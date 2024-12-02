import torch
from torch import nn
from torch.func import functional_call, grad
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split
from torchvision.datasets import *
from torchvision import transforms
from torchdiffeq import odeint
from scipy.integrate import odeint as odeint_scipy
import pytorch_lightning as pl
import os
import shutil
import numpy as np
import pandas as pd
from threading import Thread
import matplotlib.pyplot as plt

from gc_module import ContNet

print(torch.cuda.is_available())
torch.zeros(1).cuda()
torch.set_float32_matmul_precision('high')

class PredPreyDataset(Dataset):
    def __init__(self, param_file, states_dir, transform=None):
        self.param_list = pd.read_csv(param_file)
        self.states_dir = states_dir
        self.transform = transform

    def __len__(self):
        return len(self.param_list)

    def __getitem__(self, idx):
        mu = self.param_list.iloc[idx, :].to_numpy()
        param_str = '_'.join([f'{p:0.3f}' for p in mu]) + '.csv'
        state_path = os.path.join(self.states_dir, param_str)
        ut = np.loadtxt(state_path, delimiter=",", dtype=float)
        return torch.from_numpy(mu).float(), torch.from_numpy(ut).float()


class PredPreyNODE(ContNet):
    def __init__(self, logcontvar0: float, loggradvar0: float, cont_lr: float, cont_reg: float, warmup_epochs: int):
        super().__init__(logcontvar0, loggradvar0, cont_lr, cont_reg, warmup_epochs)
        self.lossfunc = F.mse_loss

        self.datasetname = 'PredPrey'
        self.epochs = 1000

        self.dataset = PredPreyDataset("predprey_params.csv", "predprey_data")
        size_train = int(len(self.dataset)*0.9)
        self.data_train, self.data_val = random_split(self.dataset, [size_train, len(self.dataset) - size_train])

        self.nx = 2
        self.T = 20.0
        self.nt = 50
        self.t_span = torch.linspace(0, self.T, self.nt, device=self.device)

        hdim = 50
        activ = nn.ReLU()
        self.net = nn.Sequential(
                                nn.Linear(self.nx + 2, hdim), activ,
                                nn.Linear(hdim, hdim), activ,
                                nn.Linear(hdim, hdim), activ,
                                nn.Linear(hdim, hdim), activ,
                                nn.Linear(hdim, self.nx)
                                )
        
        for layer in self.net.modules():
            if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def configure_optimizers(self):
        # include logcontvar in optimizer
        optimizer = torch.optim.Adam([{'params': self.net.parameters()},
                                      {'params': (self.logcontvar,), 'lr': self.cont_lr}],
                                     lr=1e-4)
        return optimizer
    
    def f(self, t, x, mu):
        xdot = self.net(torch.cat((x, mu), dim=1))
        return xdot

    def forward(self, mu):
        x0 = torch.ones((len(mu), self.nx), device=self.device)
        odefunc = lambda t_, x_: self.f(t_, x_, mu)
        xt = odeint(odefunc, x0, self.t_span.to(self.device), method='dopri5', rtol=1e-4, atol=1e-4)
        xt = torch.transpose(xt, 0, 1)
        return xt

    def training_step(self, batch, batch_idx):
        self.log('param_norm', sum(p.pow(2.0).sum() for p in self.parameters()))
        
        opt = self.optimizers()
        opt.zero_grad()

        # add gaussian noise to parameters
        self.perturb_params()
        
        # compute loss
        x, y = batch
        y_pred = self.forward(x)
        loss = self.lossfunc(y_pred, y)
        self.manual_backward(loss)

        # compute contvar gradient
        self.set_contvar_grad(self.calc_contvar_grad())

        # reload reference parameters
        self.unperturb_params()

        # perturb gradients
        self.perturb_grad()

        opt.step()
        self.log('train_loss', loss, prog_bar=True)
        #return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = self.lossfunc(y_pred, y)
        self.log('val_loss', loss, prog_bar=True)
        #return loss
    
    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=4, num_workers=4, persistent_workers=True)
        
    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=4, num_workers=4, persistent_workers=True)