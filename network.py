import torch
from torch import nn
from common import *

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.rnn1 = nn.RNN(
            input_size = 1,
            hidden_size = out_size,
            batch_first = True
        )
        self.rnn2 = nn.RNN(
            input_size = 1,
            hidden_size = out_size,
            batch_first = True
        )
        self.rnn3 = nn.RNN(
            input_size = 1,
            hidden_size = out_size,
            batch_first = True
        )
        self.rnnlist = [self.rnn1, self.rnn2, self.rnn3]

        self.conv2d = nn.Sequential(
            nn.Conv2d(3, channel, second*diff, second*diff),
            nn.ReLU(),
            nn.MaxPool2d(pool),
            nn.Conv2d(channel, len(self.rnnlist), third, third),
            nn.ReLU(),
            nn.MaxPool2d(pool)
        )

        self.norm = nn.Sequential(
            nn.Flatten(2),
            nn.BatchNorm1d(arr_size)
        )

        self.stack = nn.Sequential(
            nn.AvgPool2d((len(self.rnnlist), 1)),
            nn.Flatten(),
            nn.Linear(out_size, len(ansmap)+1)
        )
    
    def forward(self, x):
        x = torch.stack(list(map(self.conv2d, x)))
        x = self.norm(x)
        self.conv = x.detach().clone()
        x = torch.stack(
            [xi[(xi.max(1).values-xi.min(1).values).argsort()] for xi in x.transpose(1, 2)]
        ).transpose(0, 1)
        x = torch.stack(list(map(
            lambda e: self.rnnlist[e[0]](e[1].reshape((10, arr_size, -1)), None)[0], enumerate(x)))
        ).transpose(0, 1)
        self.rnn = x[:, :, -1].detach().clone()
        x = self.stack(x[:, :, -1])
        return x