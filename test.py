import torch, torchinfo
from pre_network import PreNetwork
from main_network import MainNetwork
from study import Study
from read import test_read
from common import arr_size, size, batch, lenA
from plot import plot
# print('Input Size:', batch, arr_size, lenA, size, size)
# torchinfo.summary(PreNetwork(), (batch, arr_size, lenA, size, size))
# torchinfo.summary(MainNetwork(), (batch, lenA))

load = lambda c: torch.load(f'out/model/{c}_model.pth')
pre_model, main_model = load('pre'), load('main')
study = Study(pre_model, main_model, test_read(), 750, plot(False))
study.test()