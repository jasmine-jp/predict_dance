import torch, numpy as np
from tqdm import tqdm
from common import arr_size, lenA, batch

class Study:
    def __init__(self, pre_model, main_model, read, diff, p):
        self.pre_loss = torch.nn.HuberLoss()
        self.pre_optimizer = torch.optim.RAdam(pre_model.parameters())
        self.main_loss = torch.nn.HuberLoss()
        self.main_optimizer = torch.optim.RAdam(main_model.parameters())
        self.pre_model, self.main_model, self.p = pre_model, main_model, p
        self.data, self.teach, self.plot = read
        self.diff = np.array([len(self.teach)-diff, diff])/batch

    def train(self):
        self.p.test, d = False, int(self.diff[0])
        print('train')

        for i in tqdm(range(1, d+1)):
            train, teach = self.create_randrange()

            pre_pred = self.pre_model(train)
            pre_loss = self.pre_loss(pre_pred, teach)

            self.pre_optimizer.zero_grad()
            pre_loss.backward()
            self.pre_optimizer.step()

            self.main_model.setstate('train', self.pre_model.c)
            main_pred = self.main_model(pre_pred)
            main_loss = self.main_loss(main_pred, teach)

            self.main_optimizer.zero_grad()
            main_loss.backward()
            self.main_optimizer.step()

            if (i % 300 == 0 or i == 1) and self.p.execute:
                self.p.saveimg(self.pre_model.c, self.main_model.r, teach, i)

    def test(self):
        self.p.test, d = True, int(self.diff[1])
        self.avgloss, pre_loss, mc, pc = 0, 0, 0, 0
        msum, prsum, ans = [torch.zeros(lenA) for _ in range(3)]
        print('test')

        with torch.no_grad():
            for i in tqdm(range(1, d+1)):
                train, teach = self.create_randrange()

                pre_pred = self.pre_model(train)
                pre_loss += self.pre_loss(pre_pred, teach).item()

                self.main_model.setstate('test', self.pre_model.c)
                main_pred = self.main_model(pre_pred)
                self.avgloss += self.main_loss(main_pred, teach).item()

                for m, p, t in zip(main_pred.argmax(dim=1), pre_pred.argmax(dim=1), teach):
                    mc, pc, msum[m], prsum[p], ans = mc+t[m], pc+t[p], msum[m]+1, prsum[p]+1, ans+t

                if (i % 100 == 0 or i == 1) and self.p.execute:
                    self.p.saveimg(self.pre_model.c, self.main_model.r, teach, i)

            self.avgloss, pre_loss, mc, pc = self.avgloss/d, pre_loss/d, mc/d/batch, pc/d/batch
            print(f'Main: {(100*mc):>0.1f}%, Pre: {(100*pc):>0.1f}%, Main loss: {self.avgloss:>8f}, Pre loss: {pre_loss:>8f}')
            print(f'Main: {list(map(int,msum))}, Pre: {list(map(int,prsum))}, Ans: {list(map(int,ans))}')

    def create_randrange(self):
        r = np.random.randint(0, len(self.data), batch)
        idx = np.array(list(map(lambda e: np.argmin(np.abs(self.plot-e)), r)))
        trainE = np.array(list(map(lambda e, i: e-arr_size if 0<e-self.plot[i]<arr_size else e, r, idx)))
        trainNum = np.array(list(map(lambda e: np.arange(e, e+arr_size), trainE)))
        teachNum = np.array(list(map(lambda e, i: e-(i if e < self.plot[i] else i+1)*arr_size, trainE, idx)))
        return torch.Tensor(self.data[trainNum]), torch.Tensor(self.teach[teachNum])