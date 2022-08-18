import torch
from torch import nn
from common import *
ians = range(len(ansmap)+1)
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

        self.rnn = nn.ModuleList(
            [nn.LSTM(1, hidden, batch_first=True) for _ in ians]
        )
        self.hn = nn.ParameterList([nn.Parameter(zeros) for _ in ians])

        self.stack = nn.Sequential(
            nn.AvgPool2d((len(ansmap)+1, 1)),
            nn.Flatten(),
            nn.Linear(arr_size, len(ansmap)+1)
        )

    def forward(self, x):
        self.c = torch.stack(list(map(self.conv2d, x)))
        pre = self.prestack(self.c).reshape((len(ansmap)+1,batch,-1))
        self.r = torch.stack(list(map(self.arrange,self.rnn,pre,ians))).transpose(0,1)
        return self.stack(self.r)

    def arrange(self, r, pre, i):
        o, hc = r(self.c.reshape((batch,arr_size,-1)), (self.hn[i],zeros))
        self.hn[i] = hc[0].detach().clone()
        return o[:, :, -1]*pre