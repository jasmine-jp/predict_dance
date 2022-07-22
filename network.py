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

        self.rnn = nn.LSTM(
            input_size = 1,
            hidden_size = arr_size,
            num_layers = 2,
            batch_first = True
        )

        self.stack = nn.Sequential(
            nn.Linear(arr_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, len(ansmap)+1),
            nn.Softmax(1)
        )
    
    def forward(self, x):
        x = torch.stack(list(map(lambda e: self.conv2d(e), x)))
        x = self.conv3d(x)
        x, _ = self.rnn(x, None)
        x = self.stack(x[:, -1])
        return x