import torch
from torch import nn
from common import *
ians = range(len(ansmap)+1)
zeros = torch.zeros((1, batch, hidden))

class MainNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.s = 'train'

        self.rnn = nn.ModuleList(
            [nn.LSTM(1, hidden, batch_first=True) for _ in ians]
        )
        self.hn = nn.ParameterList([nn.Parameter(zeros) for _ in ians])

        self.stack = nn.Sequential(
            nn.Linear(arr_size, len(ansmap)+1)
        )

    def forward(self, x):
        self.c = x.detach().clone()
        self.r = torch.stack(list(map(self.arrange,self.rnn,ians))).sum(0)
        return self.stack(self.r)

    def arrange(self, r, i):
        o, hn = r(self.c, (self.hn[i],zeros))
        if self.s == 'train':
            self.hn[i] = nn.Parameter(hn[0])
        return o[:, :, -1]
    
    def setstate(self, s):
        self.s = s