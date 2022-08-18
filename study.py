import torch, numpy as np
from tqdm import tqdm
from common import arr_size, batch, ans

class Study:
    def __init__(self, model, read, diff, p):
        self.loss_fn = torch.nn.HuberLoss()
        self.optimizer = torch.optim.RAdam(model.parameters())
        self.model, self.p = model, p
        self.data, self.teach, self.plot = read
        self.diff = np.array([len(self.teach)-diff, diff])/batch

    def train(self):
        print('train')
        self.p.test = False
        for i in tqdm(range(int(self.diff[0]))):
            train, teach = self.create_randrange()
            pred = self.model(train)
            loss = self.loss_fn(pred, teach)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if ((i+1) % 300 == 0 or i == 0) and self.p.execute:
                self.p.saveimg(self.model, teach, i+1)

    def test(self):
        self.test_loss, self.correct, presum, res = 0, 0, np.zeros(ans), np.array([])
        print('test')
        self.p.test = True
        with torch.no_grad():
            for i in tqdm(range(int(self.diff[1]))):
                train, teach = self.create_randrange()
                pred = self.model(train)
                self.test_loss += self.loss_fn(pred, teach).item()

                for p, t in zip(pred, teach):
                    self.correct += (t[p.argmax()] == 1).type(torch.float).sum().item()
                    presum[p.argmax()] += 1
                    res = t if res.size == 0 else res+t
                
                if ((i+1) % 100 == 0 or i == 0) and self.p.execute:
                    self.p.saveimg(self.model, teach, i+1)

        self.test_loss /= self.diff[1]
        self.correct /= self.diff[1]*batch
        print(f'Test Result: \n Accuracy: {(100*self.correct):>0.1f}%, Avg loss: {self.test_loss:>8f}')
        print(f'Sum: {list(map(int, presum))}, Ans: {list(map(int, res))}')
    
    def create_randrange(self):
        r = np.random.randint(0, len(self.data), batch)
        idx = np.array(list(map(lambda e: np.argmin(np.abs(self.plot-e)), r)))
        trainE = np.array(list(map(lambda e, i: e-arr_size if 0<e-self.plot[i]<arr_size else e, r, idx)))
        trainNum = np.array(list(map(lambda e: np.arange(e, e+arr_size), trainE)))
        teachNum = np.array(list(map(lambda e, i: e-(i if e < self.plot[i] else i+1)*arr_size, trainE, idx)))
        return torch.Tensor(self.data[trainNum]), torch.Tensor(self.teach[teachNum])