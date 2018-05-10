# -*- coding: utf-8 -*-
# network in network
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.classifer = nn.Sequential(                                     # input : None * 1 * 64 * 64
            nn.Conv2d(1, 10, kernel_size=7, stride=1, padding=2),          # Conv_1: None * 10 * 62 * 62
            nn.ReLU(inplace=True),
            nn.Conv2d(10, 10, kernel_size=1, stride=1, padding=0),        # Conv_2: None * 10 * 56 * 56
            nn.ReLU(inplace=True),
            nn.Conv2d(10, 8, kernel_size=1, stride=1, padding=0),         # Conv_3: None * 8  * 56 * 56
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),               # Maxpool: None * 8 * 28 * 28
            nn.Dropout(0.5),

            nn.Conv2d(8, 12, kernel_size=5, stride=1, padding=2),        # ConV_4: None * 12 * 28 * 28
            nn.ReLU(inplace=True),                                         # the same
            nn.Conv2d(12, 12, kernel_size=1, stride=1, padding=0),       # ConV_5: None * 12 * 28 * 28
            nn.ReLU(inplace=True),                                         # the same
            nn.Conv2d(12, 12, kernel_size=1, stride=1, padding=0),       # Conv_6: None * 12 * 28 * 28
            nn.ReLU(inplace=True),                                         # the same
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),              # Avgpool: None * 12 * 14 * 14
            nn.Dropout(0.5),

            nn.Conv2d(12, 12, kernel_size=5, stride=1, padding=0),       # None * 12 * 10 * 10
            nn.ReLU(inplace=True),
            nn.Conv2d(12, 10, kernel_size=1, stride=1, padding=0),       # None * 10 * 10 * 10
            nn.ReLU(inplace=True),
            nn.Conv2d(10, 10, kernel_size=1, stride=1, padding=0),        # None * 10 * 10 * 10
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=11, stride=1, padding=0),              # None * 10 * 1 * 1
            nn.Softmax()
        )
        # there is no FC layer in the last layers of NiN

    def forward(self, x):
        x = self.classifer(x)
        print(x.size())
        x = x.view(x.size(0), 10)
        return x