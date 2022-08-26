import torch, torchinfo
from pre_network import PreNetwork
from main_network import MainNetwork
from study import Study
from read import test_read
from common import arr_size, size
from plot import plot
# print('Input Size:', 10, arr_size, 3, size, size)
# torchinfo.summary(PreNetwork(), (10, arr_size, 3, size, size))
# torchinfo.summary(MainNetwork(), (10, arr_size, 1))

s = lambda c: f'out/model/{c}.pth'
pre_model, main_model = torch.load(s('pre_model')), torch.load(s('main_model'))
study = Study(pre_model, main_model, test_read(), 750, plot(False))

study.test()