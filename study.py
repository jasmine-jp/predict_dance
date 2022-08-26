import torch, numpy as np
from tqdm import tqdm
from common import arr_size, ansmap, batch

class Study:
    def __init__(self, pre_model, main_model, read, diff, p):
        self.pre_loss = torch.nn.BCEWithLogitsLoss()
        self.pre_optimizer = torch.optim.RAdam(pre_model.parameters())
        self.main_loss = torch.nn.SmoothL1Loss()
        self.main_optimizer = torch.optim.RAdam(main_model.parameters())
        self.pre_model, self.main_model, self.p = pre_model, main_model, p
        self.data, self.teach, self.plot = read
        self.diff = np.array([len(self.teach)-diff, diff])/batch

    def train(self):
        self.p.test = False
        print('train')

        for i in tqdm(range(int(self.diff[0]))):
            train, teach = self.create_randrange()
            self.main_model.setstate('train')

            pre_pred = self.pre_model(train)
            pre_loss = self.pre_loss(pre_pred, teach)

            self.pre_optimizer.zero_grad()
            pre_loss.backward()
            self.pre_optimizer.step()

            main_pred = self.main_model(self.pre_model.c)
            main_loss = self.main_loss(main_pred, pre_pred.detach().clone())

            self.main_optimizer.zero_grad()
            main_loss.backward()
            self.main_optimizer.step()

            if ((i+1) % 300 == 0 or i == 0) and self.p.execute:
                self.p.saveimg(self.pre_model.c, self.main_model.r, teach, i+1)

    def test(self):
        self.test_loss, self.p.test, co, d = 0, True, 0, int(self.diff[1])
        msum, prsum, ans = [torch.zeros(len(ansmap)+1) for _ in range(3)]
        print('test')

        with torch.no_grad():
            for i in tqdm(range(d)):
                train, teach = self.create_randrange()
                self.main_model.setstate('test')

                pre_pred = self.pre_model(train)
                main_pred = self.main_model(self.pre_model.c)

                self.test_loss += self.main_loss(main_pred, pre_pred.detach().clone()).item()

                for m, p, t in zip(main_pred.argmax(dim=1), pre_pred.argmax(dim=1), teach):
                    co, msum[m], prsum[p], ans = co+t[m], msum[m]+1, prsum[p]+1, ans+t

                if ((i+1) % 100 == 0 or i == 0) and self.p.execute:
                    self.p.saveimg(self.pre_model.c, self.main_model.r, teach, i+1)

            self.test_loss, co = self.test_loss/d, co/d/batch
            print(f'Accuracy: {(100*co):>0.1f}%, Avg loss: {self.test_loss:>8f}')
            print(f'Main: {list(map(int,msum))}, Pre: {list(map(int,prsum))}, Ans: {list(map(int,ans))}')

    def create_randrange(self):
        r = np.random.randint(0, len(self.data), batch)
        idx = np.array(list(map(lambda e: np.argmin(np.abs(self.plot-e)), r)))
        trainE = np.array(list(map(lambda e, i: e-arr_size if 0<e-self.plot[i]<arr_size else e, r, idx)))
        trainNum = np.array(list(map(lambda e: np.arange(e, e+arr_size), trainE)))
        teachNum = np.array(list(map(lambda e, i: e-(i if e < self.plot[i] else i+1)*arr_size, trainE, idx)))
        return torch.Tensor(self.data[trainNum]), torch.Tensor(self.teach[teachNum])