import torch
from torch import nn
from common import *
ians = range(lenA)
zeros = torch.zeros((1, batch, hidden))

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

        self.rnn = nn.ModuleList(
            [nn.LSTM(1, hidden, batch_first=True) for _ in ians]
        )
        self.hn = nn.ParameterList([nn.Parameter(zeros) for _ in ians])

        self.stack = nn.Sequential(
            nn.Linear(arr_size, lenA)
        )

    def forward(self, x):
        self.c = torch.stack(list(map(self.conv2d, x)))
        self.r = torch.stack(list(map(self.arrange, self.rnn, ians))).sum(0)
        return self.stack(self.r)

    def arrange(self, r, i):
        o, hc = r(self.c, (self.hn[i], zeros))
        self.hn[i] = nn.Parameter(hc[0])
        return o[:, :, -1]