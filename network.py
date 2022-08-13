import torch
from torch import nn
from common import *

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv2d = nn.Sequential(
            nn.Conv2d(3, channel, second*diff, second*diff),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
            nn.MaxPool2d(pool),
            nn.Conv2d(channel, 1, third, third),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(pool),
            nn.Flatten(0)
        )

        self.prestack = nn.Sequential(
            nn.Linear(arr_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, len(ansmap)+1),
            nn.Softmax(1)
        )

        self.rnn1 = nn.LSTM(
            input_size = 1,
            hidden_size = hidden
        )
        self.rnn2 = nn.LSTM(
            input_size = 1,
            hidden_size = hidden
        )
        self.rnn3 = nn.LSTM(
            input_size = 1,
            hidden_size = hidden
        )
        self.rnnlist = [self.rnn1, self.rnn2, self.rnn3]
        self.rnn1hn, self.rnn2hn, self.rnn3hn = None, None, None
        self.hnlist = [self.rnn1hn, self.rnn2hn, self.rnn3hn]

        self.stack = nn.Sequential(
            nn.AvgPool2d((1, 3)),
            nn.Flatten(),
            nn.Linear(arr_size, len(ansmap)+1)
        )
    
    def forward(self, x):
        self.c = torch.stack(list(map(self.conv2d, x)))
        pre = self.prestack(self.c).argmax(dim=1)
        self.r = torch.stack(list(map(self.arrange, pre, self.c)))
        return self.stack(self.r)
    
    def arrange(self, p, e):
        o, hn = self.rnnlist[p](e.reshape((arr_size, -1)), self.hnlist[p])
        self.hnlist[p] = (hn[0].detach().clone(), hn[1].detach().clone())
        return o[:, -1]