import torch
from torch import nn
from common import *
ians = range(lenA)
zeros = torch.zeros((1, batch, hidden))

class MainNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.s, self.c = 'train', torch.zeros((batch, arr_size, 1))

        self.rnn = nn.ModuleList(
            [nn.LSTM(1, hidden, batch_first=True) for _ in ians]
        )
        self.hn = nn.ParameterList([nn.Parameter(zeros) for _ in ians])

        self.stack = nn.Sequential(
            nn.Linear(arr_size, lenA)
        )

    def forward(self, x):
        pre = x.detach().clone().reshape((lenA, batch, 1, 1))
        self.r = torch.stack(list(map(self.arrange,self.rnn,pre,ians))).sum(0)
        return self.stack(self.r)

    def arrange(self, r, p, i):
        o, hc = r(self.c*p, (self.hn[i],zeros))
        if self.s == 'train':
            self.hn[i] = nn.Parameter(hc[0])
        return o[:, :, -1]
    
    def setstate(self, s, c):
        self.s = s
        self.c = c.detach().clone()