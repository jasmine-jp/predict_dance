import torch
from torch import nn
from common import *
ians = range(lenA)
zeros = torch.zeros((1, hidden))

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv2d = nn.Sequential(
            nn.Conv2d(3, channel, second, second),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
            nn.MaxPool2d(pool),
            nn.Conv2d(channel, 1, third*diff, third*diff),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(pool),
            nn.Flatten()
        )

        self.prestack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(arr_size, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, lenA),
            nn.Softmax(1)
        )

        self.rnn = nn.ModuleList([nn.LSTM(1, hidden) for _ in ians])
        self.hn = nn.ParameterList([nn.Parameter(zeros) for _ in ians])

        self.stack = nn.Sequential(
            nn.Linear(arr_size, lenA)
        )

    def forward(self, x):
        self.c = torch.stack(list(map(self.conv2d, x)))
        self.pre = self.prestack(self.c).argmax(dim=1)
        self.r = torch.stack(list(map(self.arrange, self.pre, self.c)))
        return self.stack(self.r)

    def arrange(self, p, e):
        o, (hn, _) = self.rnn[p](e, (self.hn[p], zeros))
        self.hn[p] = nn.Parameter(hn)
        return o[:, -1]