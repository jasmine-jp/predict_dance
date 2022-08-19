import torch
from torch import nn
from common import *
ians = range(len(ansmap)+1)
zeros = torch.zeros((1, batch, hidden))

class MainNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.rnn = nn.ModuleList(
            [nn.LSTM(1, hidden, batch_first=True) for _ in ians]
        )
        self.hn = nn.ParameterList([nn.Parameter(zeros) for _ in ians])

        self.stack = nn.Sequential(
            nn.Linear(arr_size, len(ansmap)+1)
        )

    def forward(self, x):
        p = x.detach().clone().reshape((len(ansmap)+1,batch,-1))
        self.r = torch.stack(list(map(self.arrange,self.rnn,p,ians))).sum(0)
        return self.stack(self.r)

    def arrange(self, r, p, i):
        o, hc = r(self.c.reshape((batch,arr_size,-1)), (self.hn[i],zeros))
        self.hn[i] = hc[0].detach().clone()
        return o[:, :, -1]*p
    
    def setstate(self, c):
        self.c = c.detach().clone()