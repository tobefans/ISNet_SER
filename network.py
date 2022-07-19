# -*- coding: utf-8 -*-
"""
@author: fan weiquan
"""

import torch.nn as nn
import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConvLayer(nn.Module):
    def __init__(self, input_channels, output_channels, cnn_kernel, pool_kernel, dilat=1):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=cnn_kernel, stride=1, dilation=dilat, padding=(int((cnn_kernel+(cnn_kernel-1)*(dilat-1)-1)/2))),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
            )

        self.downsample = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
            )

        if not self.pool_kernel==None:
            self.avgpool = nn.AvgPool2d(kernel_size=pool_kernel, stride=pool_kernel)

    def forward(self, x):
        out = self.conv(x)
        residual = self.downsample(x)
        out = out + residual

        if not self.pool_kernel==None:
            out = self.avgpool(out)

        return out


class AttentionTransLayer(nn.Module):
    def __init__(self, dinp, dout):
        super().__init__()
        
        self.attention = nn.Sequential(
                            nn.Linear(dinp, dinp//2),
                            nn.ReLU(),
                            nn.Linear(dinp//2, dinp),
                            nn.Sigmoid(),
                            )


        self.fc = nn.Sequential(
                            nn.Linear(dinp, dout),
                            nn.ReLU(),
                            )

    def forward(self, x):
        out = self.attention(x) * x
        out += x
        out = self.fc(x)
        return out


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = ConvLayer(3, 16, cnn_kernel=3, pool_kernel=2, dilat=1, scale=1)
        self.layer2 = ConvLayer(16, 32, cnn_kernel=3, pool_kernel=2, dilat=1, scale=1)
        self.layer3 = ConvLayer(32, 64, cnn_kernel=3, pool_kernel=2, dilat=1, scale=1)
        self.globalpooling = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        out = self.layer3(self.layer2(self.layer1(x)))
        out = self.globalpooling(out).view(x.size(0), -1)
        return out


class Translator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = AttentionTransLayer(64, 64)
        self.layer2 = AttentionTransLayer(64, 64)
        self.layer3 = AttentionTransLayer(64, 64)
        self.layer4 = AttentionTransLayer(64, 64)

    def forward(self, x):
        out = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return out


class Classifier(nn.Module):
    def __init__(self, input_dim=64, num_classes=4):
        super().__init__()
        self.linear = nn.Sequential(nn.Linear(input_dim, input_dim),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(input_dim),
                                    nn.Linear(input_dim, input_dim),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(input_dim),)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        feature = self.dropout(self.linear(x))
        out = self.classifier(feature)
        return out, feature
