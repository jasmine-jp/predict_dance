import torch, torchinfo
from network import NeuralNetwork
from study import Study
from read import test_read
from common import arr_size, size
# print('Input Size:', 10, arr_size, 3, size, size)
# torchinfo.summary(NeuralNetwork(), (10, arr_size, 3, size, size))

model = NeuralNetwork()
study = Study(model, test_read(), 10, 750)

model.load_state_dict(torch.load('out/model_weights.pth'))

study.test()