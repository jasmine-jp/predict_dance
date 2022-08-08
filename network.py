import torch
from torch import nn
from common import *

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.rnn1 = nn.GRU(
            input_size = 1,
            hidden_size = int(arr_size/1.2),
            batch_first = True
        )
        self.rnn2 = nn.GRU(
            input_size = 1,
            hidden_size = int(arr_size/1.5),
            batch_first = True
        )
        self.rnn3 = nn.LSTM(
            input_size = 1,
            hidden_size = int(arr_size/1.2),
            batch_first = True
        )
        self.rnnlist = [self.rnn1, self.rnn2, self.rnn3]

        self.conv2d = nn.Sequential(
            nn.Conv2d(3, channel, second*diff, second*diff),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
            nn.MaxPool2d(pool),
            nn.Conv2d(channel, len(self.rnnlist), third, third),
            nn.BatchNorm2d(len(self.rnnlist)),
            nn.ReLU(),
            nn.MaxPool2d(pool),
            nn.Flatten(1)
        )

        self.stack = nn.Sequential(
            nn.AvgPool2d((len(self.rnnlist), 1)),
            nn.Flatten(),
            nn.Linear(arr_size, len(ansmap)+1)
        )
    
    def forward(self, x):
        self.c = torch.stack(list(map(self.conv2d, x))).transpose(1, 2)
        self.r = torch.stack(list(map(lambda l, e:
            l(e.reshape((10, arr_size, -1)))[0][:,:,-1], self.rnnlist, torch.stack(
                [c[(c.max(1).values-c.min(1).values).argsort()] for c in self.c]
        ).transpose(0, 1)))).transpose(0, 1)
        return self.stack(self.r)