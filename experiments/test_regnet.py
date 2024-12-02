
import torch
from torch import nn
#from torch.func import functional_call, grad
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split
from torchvision import transforms, models
import pandas as pd
import shutil
import numpy as np
import os

from gc_src.gc_opt import GCOpt

torch.set_float32_matmul_precision('high')

class DemoRegDataset(Dataset):
    def __init__(self, sample_file, transform=None):
        self.sample_list = pd.read_csv(sample_file)
        self.transform = transform

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        sample = self.sample_list.iloc[idx, :].to_numpy()
        return torch.from_numpy(sample[0:-1]).float(), torch.from_numpy(sample[-1:]).float()

class RegNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.n_layers = 3
        self.h_dim = 256

        activ = nn.ReLU()
        
        layers = []
        layers.extend([nn.Linear(10, self.h_dim), activ])
        for _ in range(self.n_layers - 2):
            layers.extend([nn.Linear(self.h_dim, self.h_dim), activ])
        layers.append(nn.Linear(self.h_dim, 1))

        self.net = nn.Sequential(*layers)
        
        for layer in self.net.modules():
            if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        ypred = self.net(x)
        return ypred

if __name__ == '__main__':
    lossfunc = nn.MSELoss()

    dataset = DemoRegDataset("data/reg_data_n10_s100000.csv")
    size_train = int(len(dataset)*0.9)
    data_train, data_test = random_split(dataset, [size_train, len(dataset) - size_train])

    trainloader = DataLoader(data_train, batch_size=50, shuffle=True, num_workers=12, persistent_workers=True)
    testloader = DataLoader(data_test, batch_size=50, shuffle=False, num_workers=12, persistent_workers=True)

    mynet = RegNet()
    print(mynet)

    base_optimizer = torch.optim.Adam(mynet.parameters(), lr=1e-4, weight_decay=1e-4)
    optimizer = GCOpt(base_optimizer, cont_var0=float('nan'), cont_lr=1e-2, cont_reg=0.0)

    for epoch in range(100):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            def closure():
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward
                outputs = mynet(inputs)
                loss = lossfunc(outputs, labels)
                loss.backward()

                return loss

            loss = optimizer.step(closure)

            # print statistics
            running_loss += loss.item()
            if i % 500 == 499:    # print every 500 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 500:.3f}')
                running_loss = 0.0

    print('Finished Training')
