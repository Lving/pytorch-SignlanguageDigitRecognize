# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=7, kernel_size=9)  # 60 * 60
        self.conv2 = nn.Conv2d(in_channels=7, out_channels=10, kernel_size=7)  #
        self.fc1   = nn.Linear(11*11*10, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))  # None * 56 * 56 * 7
        out = F.max_pool2d(out, 2)  # None * 28 * 28 * 7
        out = F.relu(self.conv2(out))  # None * 22 * 22 * 10
        out = F.max_pool2d(out, 2)    # None * 11 * 11 * 10
        out = out.view(out.size(0), -1)  #
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out









