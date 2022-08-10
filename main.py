import torch
from study import Study
from network import NeuralNetwork
from read import all_read
from plot import plot

model = NeuralNetwork()
study = Study(model, all_read('video'), 5000, plot(True))

epochs = 5
for t in range(epochs):
    print(f'Epoch {t+1}\n-------------------------------')
    study.p.epoch = t+1
    study.train()
    study.test()
    print('Saving PyTorch Model State')
    torch.save(model.state_dict(), 'out/model_weights.pth')
    if study.correct > 0.9995:
        print('this prediction is perfect')
        break
print('Done!')