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

class InitCondDataset(Dataset):
    def __init__(self, transform=None):
        n_samp = 1
        self.X = torch.tensor([[np.pi, 0.0]], dtype=torch.float32)
        #self.X = torch.stack((torch.linspace(torch.pi, torch.pi*3/4, n_samp), torch.zeros(n_samp)), dim=1)
        self.transform = transform

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx,:]


class PendulumRL(ContNet):
    def __init__(self, loglambda0: float, cont_lr: float, cont_reg: float, warmup_epochs: int):
        super().__init__(loglambda0, cont_lr, cont_reg, warmup_epochs)
        self.lossfunc = F.mse_loss

        self.datasetname = 'Pendulum'
        self.epochs = 100

        self.dataset = InitCondDataset()
        #size_train = int(len(self.dataset)*0.9)
        #self.data_train, self.data_val = random_split(self.dataset, [size_train, len(self.dataset) - size_train])

        self.T = 5.0
        self.nt = 100
        self.t_span = torch.linspace(0, self.T, self.nt, device=self.device)

        self.max_torque = 2.0
        self.mass = 1.0
        self.arm_length = 1.0
        self.gravity = 9.81

        hdim = 256
        activ = nn.ReLU()
        self.net = nn.Sequential(
                                nn.Linear(2, hdim), activ,
                                nn.Linear(hdim, hdim), activ,
                                nn.Linear(hdim, 1), nn.Tanh(),
                                )
        
        for layer in self.net.modules():
            if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def configure_optimizers(self):
        # include logcontvar in optimizer
        optimizer = torch.optim.Adam([{'params': self.net.parameters()},
                                      {'params': (self.logcontvar,), 'lr': self.cont_lr}],
                                     lr=1e-3)
        return optimizer
    
    def bound_angle(self, theta):
        return ((theta + torch.pi) % (2*torch.pi)) - torch.pi

    def f(self, t, x):
        xdot = torch.zeros_like(x)
        x[:,0] = self.bound_angle(x[:,0])
        theta = x[:,0]
        theta_dot = x[:,1]
        xdot[:,0] = theta_dot

        I = self.mass*self.arm_length**2
        gravity_torque = self.mass*self.gravity*self.arm_length*torch.sin(theta)
        action_torque = self.max_torque*torch.squeeze(self.net(x))
        xdot[:,1] = (gravity_torque + action_torque)/I

        return xdot

    def forward(self, x0):
        odefunc = lambda t_, x_: self.f(t_, x_)
        xt = odeint(odefunc, x0, self.t_span.to(self.device), method='euler')
        xt = torch.transpose(xt, 0, 1)
        xt[0,:,0] = self.bound_angle(xt[0,:,0])
        return xt

    def training_step(self, batch, batch_idx):
        torch.autograd.set_detect_anomaly(True)
        self.log('param_norm', sum(p.pow(2.0).sum() for p in self.parameters()))
        
        opt = self.optimizers()
        opt.zero_grad()

        # add gaussian noise to parameters
        self.perturb_params()
        
        # compute loss
        x0 = batch
        xt = self.forward(x0)
        loss = self.lossfunc(xt[:,-1], torch.zeros_like(x0))
        self.manual_backward(loss)

        # compute contvar gradient
        self.set_contvar_grad(self.calc_contvar_grad())

        # reload reference parameters
        self.unperturb_params()

        opt.step()
        self.log('train_loss', loss, prog_bar=True)
        #return loss
    
    def validation_step(self, batch, batch_idx):
        x0 = batch
        xt = self.forward(x0)
        loss = self.lossfunc(xt[:,-1], torch.zeros_like(x0))
        self.log('val_loss', loss, prog_bar=True)
        #return loss
    
    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=1, num_workers=4, persistent_workers=True)
        
    #def val_dataloader(self):
    #    return DataLoader(self.data_val, batch_size=1, num_workers=4, persistent_workers=True)