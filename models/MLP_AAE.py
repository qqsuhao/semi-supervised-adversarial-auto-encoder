# -*- coding:utf8 -*-
# @TIME     : 2020/10/30 14:50
# @Author   : Hao Su
# @File     : MLP_AAE.py

'''
reference: https://github.com/andreandradecosta/pytorch_aae
'''

import torch.nn as nn
import torch

class Encoder(nn.Module):
    def __init__(self, imageSize, z_dim, n_classes):
        super(Encoder, self).__init__()
        self.imageSize = imageSize
        self.z_dim = z_dim
        self.n_classes = n_classes
        self.base_model = nn.Sequential(
            nn.Linear(self.imageSize, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
        )
        self.latent = nn.Linear(1000, self.z_dim)
        self.cat = nn.Linear(1000, self.n_classes)


    def forward(self, x):
        x = x.view(-1, self.imageSize)
        y = self.base_model(x)
        y_latent = self.latent(y)
        y_cat = self.cat(y)
        return y_latent, y_cat


class Decoder(nn.Module):
    def __init__(self, imageSize, z_dim, n_classes):
        super(Decoder, self).__init__()
        self.imageSize = imageSize
        self.z_dim = z_dim
        self.n_classes = n_classes
        self.model = nn.Sequential(
            nn.Linear(self.z_dim + self.n_classes, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, self.imageSize),
            nn.Sigmoid()
        )


    def forward(self, x):
        y = self.model(x)
        return y


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1),
            nn.Sigmoid(),
        )


    def forward(self, x):
        y = self.model(x)
        return y


def print_net():
    try:
        from torchsummary import summary
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        enc = Encoder(28**2, 10, 10).to(device)
        dec = Decoder(28**2, 10, 10).to(device)
        disc = Discriminator(10).to(device)
        summary(enc, (1, 28**2))
        summary(dec, (1, 20))
        summary(disc, (1, 10))
    except:
        print("No Module Find for Summary")


# print_net()