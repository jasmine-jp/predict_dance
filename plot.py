import matplotlib.pyplot as plt

class plot:
    def __init__(self, execute):
        self.epoch = 0
        self.test = False
        self.execute = execute
    
    def saveimg(self, model, ans, idx):
        conv, rnn = model.c[0].detach().clone(), model.r[0].detach().clone()
        fig = plt.figure(figsize=(12.8, 4.8))
        color = ['tab:blue', 'tab:orange', 'tab:green']
        name = list(map(lambda e : str(type(e)).split('.')[-1][:-2], model.rnnlist))
        value = (conv.max(1).values-conv.min(1).values).argsort()
        fig.suptitle(f'{ans[0]}')
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        ax1.set_title('conv')
        ax2.set_title('rnn')
        for i, (c, r) in enumerate(zip(conv, rnn)):
            labelname = f'{name[value[i]]}:{["min","mid","max"][i]}'
            ax1.plot(list(map(float, c)), c=color[i])
            ax2.plot(list(map(float, r)), c=color[value[i]], label=labelname)
        ax2.legend(bbox_to_anchor=(1, 1), frameon=False)
        s = 'test' if self.test else 'epoch_'+str(self.epoch)
        fig.savefig(f'img/{s}/estimate_{idx}')