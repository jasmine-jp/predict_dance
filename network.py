import torch
from torch import nn
from common import *
hidden = 10

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv2d = nn.Sequential(
            nn.Conv2d(3, channels, kernel, stride),
            nn.ReLU(),
            nn.AvgPool2d(pool),
            nn.Flatten()
        )

        self.cnn = nn.LSTM(
            input_size = int(node/arr_size),
            hidden_size = hidden,
            batch_first = True
        )

        self.stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(arr_size*hidden, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, len(ansmap)+1),
            nn.Softmax(1)
        )
    
    def forward(self, x):
        x = torch.stack(list(map(lambda e: self.conv2d(e), x)))
        x, _ = self.cnn(x, None)
        x = self.stack(x)
        return x