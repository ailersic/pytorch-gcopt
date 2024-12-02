
import torch
from torch import nn
from torch.func import functional_call, grad
from torch.nn import functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import CIFAR10
from torchvision import transforms, models
import pytorch_lightning as pl
import os

from gc_src.gc_hyperopt import GCHyperOptimizer
from gc_src.gc_opt import GCOptimizer

torch.set_float32_matmul_precision('high')


class CIFAR10Class(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = models.resnet50()
        self.net.fc = nn.Linear(self.net.fc.in_features, 10)

        '''
        activ = nn.ReLU()
        self.net = nn.Sequential(
                                nn.Conv2d(3, 64, 3, padding=1), activ,
                                nn.Conv2d(64, 32, 3, padding=1), activ,
                                nn.Conv2d(32, 16, 3, padding=1), activ,
                                nn.Conv2d(16, 8, 3, padding=1), activ,
                                nn.Flatten(start_dim=1),
                                nn.Linear(8192, 256),
                                nn.Softmax(dim=1)
                                )'''
        
        for layer in self.net.modules():
            if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
                #nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        return self.net(x)

if __name__ == '__main__':
    lossfunc = F.cross_entropy

    dataset = CIFAR10(os.getcwd(), download=True, transform=transforms.ToTensor())
    size_train = int(len(dataset)*0.9)
    data_train, data_test = random_split(dataset, [size_train, len(dataset) - size_train])

    trainloader = DataLoader(data_train, batch_size=50, shuffle=True, num_workers=12, persistent_workers=True)
    testloader = DataLoader(data_test, batch_size=50, shuffle=False, num_workers=12, persistent_workers=True)

    mynet = CIFAR10Class()
    base_optimizer = optim.Adam(mynet.parameters(), lr=1e-4)
    
    #optimizer = base_optimizer
    #optimizer = GCOptimizer(base_optimizer, cont_var0=1e-4, cont_lr=1e-1)
    optimizer = GCHyperOptimizer(base_optimizer, contvar0=torch.tensor([1e-6]), contvar_lr=1e0)

    for epoch in range(20):  # loop over the dataset multiple times

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
            if i % 10 == 9:    # print every 500 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 500:.3f} contvar: {torch.exp(optimizer.logcontvar).item():.3E}')
                running_loss = 0.0

    print('Finished Training')