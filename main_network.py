import torch
from torch import nn
from common import *
ians = range(lenA)
zeros = torch.zeros((1, batch, hidden))

class MainNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.hb,self.cb,self.c = False,False,torch.zeros((batch,arr_size,1))

        self.rnn = nn.ModuleList(
            [nn.LSTM(1, hidden, batch_first=True) for _ in ians]
        )
        self.hn = nn.ParameterList([nn.Parameter(zeros, False) for _ in ians])
        self.cn = nn.ParameterList([nn.Parameter(zeros, False) for _ in ians])

        self.stack = nn.Sequential(
            nn.Linear(arr_size, lenA)
        )

    def forward(self, x):
        self.c = x.detach().clone()
        self.r = torch.stack(list(map(self.arrange,self.rnn,ians))).sum(0)
        self.hb, self.cb = False, False
        return self.stack(self.r)

    def arrange(self, r, i):
        o, hc = r(self.c, (self.hn[i],self.cn[i]))
        self.hn[i].data = hc[0] if self.hb else self.hn[i]
        self.cn[i].data = hc[1] if self.cb else self.cn[i]
        return o[:, :, -1]

    def setstate(self, loss=1.0):
        self.hb, self.cb = loss < 0.005, loss < 0.001