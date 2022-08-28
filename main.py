import torch
from study import Study
from pre_network import PreNetwork
from main_network import MainNetwork
from read import all_read
from plot import plot

pre_model, main_model = PreNetwork(), MainNetwork()
study = Study(pre_model, main_model, all_read('video'), 5000, plot(True))
loss = 1

epochs = 20
for t in range(epochs):
    print(f'Epoch {t+1}\n-------------------------------')
    study.p.epoch = t+1
    study.train()
    study.test()
    if loss > study.avgloss:
        print('Saving PyTorch Model State')
        torch.save(pre_model, 'out/model/pre_model.pth')
        torch.save(main_model, 'out/model/main_model.pth')
        loss = study.avgloss
print(f'final loss: {loss}')