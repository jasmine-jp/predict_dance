import torch
from study import Study
from network import NeuralNetwork
from read import all_read

model = NeuralNetwork()
read = all_read('video')
study = Study(model, read, 10, 5000)

epochs = 3
for t in range(epochs):
    print(f'Epoch {t+1}\n-------------------------------')
    study.train()
    study.test()
    print('Saving PyTorch Model State')
    torch.save(model.state_dict(), 'out/model_weights.pth')
    if study.correct > 0.9995:
        print('this prediction is perfect')
        break
print('Done!')