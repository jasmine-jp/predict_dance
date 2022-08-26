import torch
from study import Study
from network import NeuralNetwork
from read import all_read
from plot import plot

model = NeuralNetwork()
study = Study(model, all_read('video'), 5000, plot(True))
loss = 1

epochs = 20
for t in range(epochs):
    print(f'Epoch {t+1}\n-------------------------------')
    study.p.epoch = t+1
    study.train()
    study.test()
    if loss > study.test_loss:
        print('Saving PyTorch Model State')
        torch.save(model, 'out/model/model_weights.pth')
        loss = study.test_loss
print(f'final loss: {loss}')