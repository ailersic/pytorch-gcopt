import torch
from torch import Tensor
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import CelebA
from torchvision import transforms, models
from torchmetrics.image.fid import FrechetInceptionDistance
#from ignite.metrics.gan import FID

import pytorch_lightning as pl
import os

from gc_module import ContNet

torch.set_float32_matmul_precision('high')

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 128

# Size of feature maps in discriminator
ndf = 64

# Generator Code
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. ``(ngf*4) x 8 x 8``
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. ``(ngf*2) x 16 x 16``
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. ``(ngf) x 32 x 32``
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1),
            nn.Sigmoid()
            # state size. ``(nc) x 64 x 64``
        )

    def forward(self, input):
        return self.main(input)

# Discriminator Code
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is ``(nc) x 64 x 64``
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 32 x 32``
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 8 x 8``
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 4 x 4``
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
            # state size. ``(1) x 1 x 1``
            nn.Flatten()
        )

    def forward(self, input):
        return self.main(input)

class CelebAGAN(ContNet):
    def __init__(self, logcontvar0: float, loggradvar0: float, cont_lr: float, cont_reg: float, warmup_epochs: int):
        super().__init__(logcontvar0, loggradvar0, cont_lr, cont_reg, warmup_epochs)

        self.fid = FrechetInceptionDistance(feature=64, reset_real_features=False, normalize=True)
        self.fid.set_dtype(torch.float32)

        self.criterion = nn.BCELoss()
        self.datasetname = 'CelebAGAN'
        self.epochs = 20

        transformlist = transforms.Compose([
            transforms.Resize(size=64),
            transforms.CenterCrop(size=(64, 64)),
            transforms.ToTensor(),
        ])
        self.dataset = CelebA(os.getcwd(), download=True, target_type='attr', transform=transformlist)

        self.G = Generator()
        self.D = Discriminator()

        for layer in self.G.modules():
            if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        for layer in self.D.modules():
            if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        # Important: This property activates manual optimization.
        self.automatic_optimization = False

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=128, shuffle=True, num_workers=16, persistent_workers=True)

    def sample_z(self, n) -> Tensor:
        sample = torch.randn((n, nz, 1, 1), device=self.device)
        return sample

    def sample_G(self, n) -> Tensor:
        z = self.sample_z(n)
        return self.G(z)

    def training_step(self, batch, batch_idx):
        g_opt, d_opt, cont_opt = self.optimizers()

        X, _ = batch
        batch_size = X.shape[0]

        real_label = torch.ones((batch_size, 1), device=self.device)
        fake_label = torch.zeros((batch_size, 1), device=self.device)

        # perturb parameters for continuation
        self.perturb_params()

        g_X = self.sample_G(batch_size)

        # calculate FID
        try:
            self.fid.reset()
            self.fid.update(X, real=True)
            self.fid.update(g_X, real=False)
            self.log_dict({"fid": self.fid.compute()}, prog_bar=True)
        except:
            self.fid = FrechetInceptionDistance(feature=64, reset_real_features=False, normalize=True)
            self.log_dict({"fid": torch.nan}, prog_bar=True)

        ######################
        # Optimize Generator #
        ######################
        d_z = self.D(g_X)
        errG = self.criterion(d_z, real_label)

        g_opt.zero_grad()
        self.manual_backward(errG)

        contvar_grad = self.calc_contvar_grad()
        self.unperturb_params()
        g_opt.step()

        self.log_dict({"g_loss": errG}, prog_bar=True)

        ##########################
        # Optimize Discriminator #
        ##########################
        #self.reperturb_params()

        d_x = self.D(X)
        errD_real = self.criterion(d_x, real_label)

        d_z = self.D(g_X.detach())
        errD_fake = self.criterion(d_z, fake_label)

        errD = errD_real + errD_fake

        d_opt.zero_grad()
        self.manual_backward(errD)
        #self.unperturb_params()
        d_opt.step()

        ####################
        # Optimize ContVar #
        ####################
        cont_opt.zero_grad()
        self.set_contvar_grad(contvar_grad)
        cont_opt.step()

        self.log_dict({"d_loss": errD}, prog_bar=True)

    def configure_optimizers(self):
        g_opt = torch.optim.Adam(self.G.parameters(), lr=1e-4)
        d_opt = torch.optim.Adam(self.D.parameters(), lr=2e-4)
        cont_opt = torch.optim.Adam((self.logcontvar, ), lr=self.cont_lr)
        return g_opt, d_opt, cont_opt