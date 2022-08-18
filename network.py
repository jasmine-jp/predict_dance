import torch
from torch import nn
from common import *

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
            nn.Flatten(0)
        )

        self.prestack = nn.Sequential(
            nn.Linear(arr_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, ans),
            nn.Softmax(1)
        )

        self.rnnlist = nn.ModuleList([nn.LSTM(1, hidden) for _ in range(ans)])
        self.hn = nn.ParameterList([nn.Parameter(torch.zeros((1, hidden))) for _ in range(ans)])
        self.cn = nn.ParameterList([nn.Parameter(torch.zeros((1, hidden))) for _ in range(ans)])

        self.stack = nn.Sequential(
            nn.Linear(arr_size, ans)
        )

    def forward(self, x):
        self.c = torch.stack(list(map(self.conv2d, x)))
        pre = self.prestack(self.c).argmax(dim=1)
        self.r = torch.stack(list(map(self.arrange, pre, self.c)))
        return self.stack(self.r)

    def arrange(self, p, e):
        o, hc = self.rnnlist[p](e.reshape((arr_size, -1)), (self.hn[p], self.cn[p]))
        self.hn[p], self.cn[p] = map(lambda e: e.detach().clone(), hc)
        return o[:, -1]