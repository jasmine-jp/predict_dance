import torch
from torch import nn
from common import *

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv2d = nn.Sequential(
            nn.Conv2d(3, channel, second*diff, second*diff),
            nn.ReLU(),
            nn.MaxPool2d(pool)
        )

        self.conv3d = nn.Sequential(
            nn.Conv3d(arr_size, arr_size, third, third),
            nn.ReLU(),
            nn.MaxPool3d(pool),
            nn.Flatten(2)
        )

        self.rnndict = {
            key: nn.RNN(
                input_size = 1,
                hidden_size = 64
            ) for key in ansmap.keys()
        }

        self.stack = nn.Sequential(
            nn.Linear(64, len(ansmap))
        )
    
    def forward(self, x, teach):
        x = torch.stack(list(map(self.conv2d, x)))
        x = self.conv3d(x)
        x = torch.stack(list(map(self.rnniter, x, teach)))
        x = self.stack(x[:, -1])
        return x
    
    def rnniter(self, e, t):
        kind = list(ansmap.keys())[list(ansmap.values()).index(list(t))]
        x, _ = self.rnndict[kind](e, None)
        return x