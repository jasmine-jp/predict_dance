import torch, torchinfo
from network import NeuralNetwork
from study import Study
from read import test_read
from common import arr_size, size
from plot import plot
# print('Input Size:', 10, arr_size, 3, size, size)
# torchinfo.summary(NeuralNetwork(), (10, arr_size, 3, size, size))

model = NeuralNetwork()
study = Study(model, test_read(), 750, plot(False))

model.load_state_dict(torch.load('out/model_weights.pth'))

study.test()