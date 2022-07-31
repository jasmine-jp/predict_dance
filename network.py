import torch
from torch import nn
from common import *

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.rnn1 = nn.RNN(
            input_size = 1,
            hidden_size = out_size
        )
        self.rnn2 = nn.RNN(
            input_size = 1,
            hidden_size = out_size
        )
        self.rnn3 = nn.RNN(
            input_size = 1,
            hidden_size = out_size
        )
        self.rnnlist = [self.rnn1, self.rnn2, self.rnn3]

        self.conv2d = nn.Sequential(
            nn.Conv2d(3, channel, second*diff, second*diff),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
            nn.MaxPool2d(pool),
            nn.Conv2d(channel, len(self.rnnlist), third, third),
            nn.BatchNorm2d(len(self.rnnlist)),
            nn.ReLU(),
            nn.MaxPool2d(pool),
            nn.Flatten(1)
        )

        self.stack = nn.Sequential(
            nn.MaxPool2d((len(self.rnnlist), 1)),
            nn.Flatten(),
            nn.Linear(out_size, len(ansmap)+1)
        )
    
    def forward(self, x):
        x = torch.stack(list(map(self.conv2d, x)))
        x = torch.stack([torch.stack(list(map(lambda e, rnn: rnn(e.reshape(arr_size, -1), None)[0], \
            [xi[:, i] for i in range(len(self.rnnlist))], self.rnnlist))) for xi in x])
        x = self.stack(x[:, :, -1])
        return x