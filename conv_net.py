# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable

from sklearn.model_selection import train_test_split


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=9)  # 60 * 60
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=7)  #
        self.fc1   = nn.Linear(11*11*6, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))  # None * 56 * 56 * 3
        out = F.max_pool2d(out, 2)  # None * 28 * 28 * 3
        out = F.relu(self.conv2(out))  # None * 22 * 22 * 6
        out = F.max_pool2d(out, 2)    # None * 11 * 11 * 6
        out = out.view(out.size(0), -1)  #
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class SignLanguageDataSet(Dataset):
    """ sign language dataset"""

    def __init__(self, X, Y, onehot=True):
        """
        :param X_file:
        :param Y_file:
        """
        X = X.reshape((-1, 1, 64, 64))  # 64 * 64 add a channel
        if onehot:
            Y = np.argmax(Y, axis=1)

        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        X = self.X[idx]
        Y = self.Y[idx]

        sample = {'X': X, 'Y': Y}
        return sample


X = np.load('sign-data/X.npy')
Y = np.load('sign-data/Y.npy')
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

trainSignData = SignLanguageDataSet(x_train, y_train, onehot=True)
trainDataLoader = DataLoader(trainSignData, shuffle=True, batch_size=32)

testSignData = SignLanguageDataSet(x_test, y_test, onehot=True)
testDataLoader = DataLoader(testSignData, shuffle=False, batch_size=32)

net = LeNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, sample in enumerate(trainDataLoader):
        optimizer.zero_grad()
        inputs, targets = Variable(sample['X']), Variable(sample['Y'])
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        print('TRAIN Epoch %d | Loss: %.3f | Acc: %.3f%% (%d/%d)' % (epoch, train_loss/(batch_idx+1), 100.*correct/total,
                                                               correct, total))


def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, sample in enumerate(testDataLoader):
        inputs, targets = Variable(sample['X'], volatile=True), Variable(sample['Y'])
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        print('TEST Epoch %d | Loss: %.3f | Acc: %.3f%% (%d/%d)' % (epoch, test_loss / (batch_idx + 1), 100.*correct/total,
                                                               correct, total))


if __name__ == '__main__':
    for epoch in range(start_epoch, start_epoch+200):
        train(epoch)
        test(epoch)







