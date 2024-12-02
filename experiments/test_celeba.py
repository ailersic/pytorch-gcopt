
import torch
from torch import nn
#from torch.func import functional_call, grad
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split
from torchvision.datasets import CelebA
from torchvision import transforms, models
import pandas as pd
import shutil
import numpy as np
import os

from gc_src.gc_opt import GCOpt

print(torch.cuda.is_available())
torch.zeros(1).cuda()

torch.set_float32_matmul_precision('high')

if __name__ == '__main__':
    lossfunc = F.cross_entropy

    dataset = CelebA(os.getcwd(), download=True, target_type='attr', transform=transforms.ToTensor())
    size_train = int(len(dataset)*0.9)
    data_train, data_test = random_split(dataset, [size_train, len(dataset) - size_train])

    trainloader = DataLoader(data_train, batch_size=128, shuffle=True, num_workers=12, persistent_workers=True)
    testloader = DataLoader(data_test, batch_size=128, shuffle=False, num_workers=12, persistent_workers=True)

    mynet = models.resnet50()
    mynet.fc = nn.Linear(mynet.fc.in_features, 40)
    print(mynet)

    base_optimizer = torch.optim.Adam(mynet.parameters(), lr=3e-4)
    optimizer = GCOpt(base_optimizer, cont_var0=float('nan'), cont_lr=1e-2, cont_reg=0.0)

    for epoch in range(10):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            def closure():
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward
                outputs = mynet(inputs)
                loss = lossfunc(outputs, labels.to(torch.float))
                loss.backward()

                return loss

            loss = optimizer.step(closure)

            # print statistics
            running_loss += loss.item()
            if i % 5 == 4:    # print every n mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 5:.3f}')
                running_loss = 0.0

    print('Finished Training')
