import torch, numpy as np
from tqdm import tqdm
from common import arr_size, ansmap, batch

class Study:
    def __init__(self, pre_model, main_model, read, diff, p):
        self.pre_loss = torch.nn.BCELoss()
        self.pre_optimizer = torch.optim.RAdam(pre_model.parameters())
        self.main_loss = torch.nn.HuberLoss()
        self.main_optimizer = torch.optim.RAdam(main_model.parameters())
        self.pre_model, self.main_model, self.p = pre_model, main_model, p
        self.data, self.teach, self.plot = read
        self.diff = np.array([len(self.teach)-diff, diff])/batch

    def train(self):
        print('train')
        self.p.test = False
        for i in tqdm(range(int(self.diff[0]))):
            train, teach = self.create_randrange()

            pre_pred = self.pre_model(train)
            self.main_model.setstate(self.pre_model.c)
            pre_loss = self.pre_loss(pre_pred, teach)

            self.pre_optimizer.zero_grad()
            pre_loss.backward()
            self.pre_optimizer.step()

            main_pred = self.main_model(pre_pred)
            main_loss = self.main_loss(main_pred, teach)

            self.main_optimizer.zero_grad()
            main_loss.backward()
            self.main_optimizer.step()

            if ((i+1) % 300 == 0 or i == 0) and self.p.execute:
                self.p.saveimg(self.pre_model.c, self.main_model.r, teach, i+1)

    def test(self):
        self.test_loss, self.correct, psum, ans = 0, 0, np.zeros(len(ansmap)+1), np.array([])
        print('test')
        self.p.test = True
        with torch.no_grad():
            for i in tqdm(range(int(self.diff[1]))):
                train, teach = self.create_randrange()

                pre_pred = self.pre_model(train)
                self.main_model.setstate(self.pre_model.c)
                main_pred = self.main_model(pre_pred)

                self.test_loss += self.main_loss(main_pred, teach).item()

                for p, t in zip(main_pred, teach):
                    self.correct += (t[p.argmax()] == 1).type(torch.float).sum().item()
                    psum[p.argmax()] += 1
                    ans = t if ans.size == 0 else ans+t
                
                if ((i+1) % 100 == 0 or i == 0) and self.p.execute:
                    self.p.saveimg(self.pre_model.c, self.main_model.r, teach, i+1)

        self.test_loss /= self.diff[1]
        self.correct /= self.diff[1]*batch
        print(f'Test Result: \n Accuracy: {(100*self.correct):>0.1f}%, Avg loss: {self.test_loss:>8f}')
        print(f'Sum: {list(map(int, psum))}, Ans: {list(map(int, ans))}')
    
    def create_randrange(self):
        r = np.random.randint(0, len(self.data), batch)
        idx = np.array(list(map(lambda e: np.argmin(np.abs(self.plot-e)), r)))
        trainE = np.array(list(map(lambda e, i: e-arr_size if 0<e-self.plot[i]<arr_size else e, r, idx)))
        trainNum = np.array(list(map(lambda e: np.arange(e, e+arr_size), trainE)))
        teachNum = np.array(list(map(lambda e, i: e-(i if e < self.plot[i] else i+1)*arr_size, trainE, idx)))
        return torch.Tensor(self.data[trainNum]), torch.Tensor(self.teach[teachNum])