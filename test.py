import torch
from network import NeuralNetwork
from study import Study
from read import test_read
import torchinfo
from common import arr_size, size

read = test_read()

model = NeuralNetwork()
#torchinfo.summary(model, (10, arr_size, 3, size, size))
study = Study(model, read, 10, 750)


model.load_state_dict(torch.load('out/model_weights.pth'))

study.test()