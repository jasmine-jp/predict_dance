import torch
from tqdm import tqdm
import numpy as np
from common import arr_size, ansmap

class Study:
    def __init__(self, model, read, batch, diff):
        self.loss_fn = torch.nn.HuberLoss()
        self.optimizer = torch.optim.Adam(model.parameters())
        self.model, self.batch = model, batch
        self.data, self.teach, self.plot = read
        self.diff = np.array([len(self.teach)-diff, diff])/self.batch

    def train(self):
        print('train')
        for _ in tqdm(range(int(self.diff[0]))):
            train, teach = self.create_randrange()
            pred = self.model(train, teach)
            loss = self.loss_fn(pred, teach)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def test(self):
        test_loss, self.correct, presum, ans = 0, 0, np.zeros(len(ansmap)), np.array([])
        print('test')
        with torch.no_grad():
            for _ in tqdm(range(int(self.diff[1]))):
                train, teach = self.create_randrange()
                pred = self.model(train, teach)
                test_loss += self.loss_fn(pred, teach).item()

                for p, t in zip(pred, teach):
                    self.correct += (t[p.argmax()] == 1).type(torch.float).sum().item()
                    presum[p.argmax()] += 1
                    ans = t if ans.size == 0 else ans+t

        test_loss /= self.diff[1]
        self.correct /= self.diff[1]*self.batch
        print(f'Test Result: \n Accuracy: {(100*self.correct):>0.1f}%, Avg loss: {test_loss:>8f}')
        print(f'Sum: {list(map(int, presum))}, Ans: {list(map(int, ans))}')
    
    def create_randrange(self):
        r = np.random.randint(0, len(self.data), self.batch)
        idx = np.array(list(map(lambda e: np.argmin(np.abs(self.plot-e)), r)))
        trainE = np.array(list(map(lambda e, i: e-arr_size if 0<e-self.plot[i]<arr_size else e, r, idx)))
        trainNum = np.array(list(map(lambda e: np.arange(e, e+arr_size), trainE)))
        teachNum = np.array(list(map(lambda e, i: e-(i if e < self.plot[i] else i+1)*arr_size, trainE, idx)))
        return torch.Tensor(self.data[trainNum]), torch.Tensor(self.teach[teachNum])