import torch
from study import Study
from network import NeuralNetwork
from read import all_read
from plot import plot

model = NeuralNetwork()
p = plot(True)
study = Study(model, all_read('video'), 10, 5000, p)

epochs = 3
for t in range(epochs):
    print(f'Epoch {t+1}\n-------------------------------')
    p.epoch, p.test = t, False
    study.train()
    study.test()
    print('Saving PyTorch Model State')
    torch.save(model.state_dict(), 'out/model_weights.pth')
    if study.correct > 0.9995:
        print('this prediction is perfect')
        break
print('Done!')