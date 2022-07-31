import torch
from torch import nn
from common import *

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv2d = nn.Sequential(
            nn.Dropout2d(0.8),
            nn.Conv2d(3, channel, second*diff, second*diff),
            nn.ReLU(),
            nn.MaxPool2d(pool)
        )

        self.conv3d = nn.Sequential(
            nn.Conv3d(arr_size, arr_size, third, third),
            nn.BatchNorm3d(arr_size),
            nn.ReLU(),
            nn.MaxPool3d(pool),
            nn.Flatten(2)
        )

        self.rnn = nn.LSTM(
            input_size = 1,
            hidden_size = 64,
            batch_first = True
        )

        self.stack = nn.Sequential(
            nn.Dropout(0.8),
            nn.Linear(64, len(ansmap)+1)
        )
    
    def forward(self, x):
        x = torch.stack(list(map(self.conv2d, x)))
        x = self.conv3d(x)
        x, _ = self.rnn(x, None)
        x = self.stack(x[:, -1])
        return x