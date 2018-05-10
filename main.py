# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable

from sklearn.model_selection import train_test_split

# from conv_origin import Net
from nin import Net


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

net = Net()
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
    for epoch in range(start_epoch, start_epoch+20):
        train(epoch)
    test(epoch)







