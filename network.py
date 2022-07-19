import torch
from torch import nn
from common import arr_size, channels, kernel, stride, pool, ansmap

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv2d = nn.Sequential(
            nn.Conv2d(3, channels, kernel, stride),
            nn.ReLU(),
            nn.MaxPool2d(pool)
        )

        self.conv3d = nn.Sequential(
            nn.Conv3d(arr_size, channels, 2, 2),
            nn.ReLU()
        )

        self.stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1250, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, len(ansmap)+1),
            nn.Softmax(1)
        )
    
    def forward(self, x):
        x = torch.stack(list(map(lambda e: self.conv2d(e), x)))
        x = self.conv3d(x)
        x = self.stack(x)
        return x