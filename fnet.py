# -*- coding: utf-8 -*-
from PIL import Image,ImageChops,ImageEnhance
import torch
import os
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from torchvision.transforms import RandomResizedCrop
import cv2
from torch.nn import Conv2d
from torch import nn
from torchvision import datasets, transforms,models

class barmodel2(nn.Module):

    def __init__(self,  num_classes=1000):
        super(barmodel, self).__init__()
        # self.bnet = nn.Sequential(
        self.conv = nn.Conv2d(2, 10, kernel_size=2, padding=1,bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(25000, num_classes, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1,x2):
        x1 = x1[:,0:1,:,:]
        x2 = x2[:, 0:1, :, :]
        x = torch.cat((x1, x2), dim=1)
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        # print(x.size())
        # assert 1>2
        x = self.classifier(x)
        x = self.sigmoid(x)
        return x

class SigmodLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(SigmodLinear, self).__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        return self.sigmod(self.linear(x))

class barmodel(nn.Module):
    def __init__(self, in_features, out_features,pretrained=True):
        super(barmodel, self).__init__()
        self.net = models.mobilenet_v2(pretrained=pretrained)
        self.net.classifier[1] = SigmodLinear(in_features, out_features)
        self.net.features[0][0] = Conv2d(2, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        # self.net.features[0] = newConvNormActivation(in_channels=6,out_channels=32,kernel_size=(3, 3), stride=(2, 2),padding=(1, 1))

    def forward(self, x1,x2):
        x1 = x1[:, 0:1, :, :]
        x2 = x2[:, 0:1, :, :]
        x = torch.cat((x1,x2),dim=1)
        return self.net(x)