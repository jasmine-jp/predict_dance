import torch, torchinfo
from pre_network import PreNetwork
from main_network import MainNetwork
from study import Study
from read import test_read
from common import arr_size, size
from plot import plot
# print('Input Size:', 10, arr_size, 3, size, size)
# torchinfo.summary(PreNetwork(), (10, arr_size, 3, size, size))

pre_model, main_model = PreNetwork(), MainNetwork()
study = Study(pre_model, main_model, test_read(), 750, plot(False))

pre_model.load_state_dict(torch.load('out/pre_model.pth'))
main_model.load_state_dict(torch.load('out/main_model.pth'))

study.test()