import torch
from study import Study
from network import NeuralNetwork
from read import all_read
from plot import plot

model = NeuralNetwork()
study = Study(model, all_read('video'), 5000, plot(True))
correct = 0

epochs = 20
for t in range(epochs):
    print(f'Epoch {t+1}\n-------------------------------')
    study.p.epoch = t+1
    study.train()
    study.test()
    if correct < study.correct:
        print('Saving PyTorch Model State')
        torch.save(model.state_dict(), 'out/model_weights.pth')
        correct = study.correct
    if study.correct > 0.9995:
        print('this prediction is perfect')
        break
print('Done!')