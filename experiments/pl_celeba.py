
import torch
from torch import nn
from torch.func import functional_call, grad
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import CelebA
from torchvision import transforms, models
import pytorch_lightning as pl
import os

from gc_src.gc_opt import GCOptimizer, WarmupScheduler

torch.set_float32_matmul_precision('high')


class CelebAClass(pl.LightningModule):
    def __init__(self, cont_var0=float('nan'), cont_lr=1e-2, cont_reg=0.0, warmup_epochs=0):
        super().__init__()

        self.cont_var0 = cont_var0
        self.cont_lr = cont_lr
        self.cont_reg = cont_reg
        self.warmup_epochs = warmup_epochs

        self.lossfunc = F.mse_loss#F.cross_entropy

        self.datasetname = 'CelebA'
        self.epochs = 20

        self.dataset = CelebA(os.getcwd(), download=True, target_type='landmarks', transform=transforms.ToTensor())
        size_train = int(len(self.dataset)*0.9)
        self.data_train, self.data_val = random_split(self.dataset, [size_train, len(self.dataset) - size_train])

        self.net = models.resnet18()
        self.net.fc = nn.Linear(self.net.fc.in_features, 10)
        
        for layer in self.net.modules():
            if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        ypred = self.net(x)
        return ypred
    
    def configure_optimizers(self):
        # include logcontvar in optimizer
        base_optimizer = torch.optim.Adam(self.net.parameters(), lr=3e-4)
        optimizer = GCOptimizer(base_optimizer, cont_var0=self.cont_var0, cont_lr=self.cont_lr, cont_reg=self.cont_reg)
        scheduler = WarmupScheduler(optimizer, warmup_epochs=self.warmup_epochs)

        return [optimizer], [scheduler]

    def training_step(self, train_batch, batch_idx):
        self.log('param_norm', sum(p.pow(2.0).sum() for p in self.parameters()))
        opt = self.optimizers()
        self.log('contvar', torch.exp(opt.logcontvar).detach(), prog_bar=True)
        
        # compute loss
        x, y = train_batch
        ypred = self.forward(x)
        loss = self.lossfunc(ypred, y.to(torch.float))
        self.log('train_loss', loss.detach(), prog_bar=True)
        
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        ypred = self.net(x)

        loss = self.lossfunc(ypred, y.to(torch.float))
        self.log('val_loss', loss.detach(), prog_bar=True)
        
        return loss
    
    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=128, shuffle=True, num_workers=16, persistent_workers=True)
        
    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=128, num_workers=16, persistent_workers=True)
