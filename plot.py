import matplotlib.pyplot as plt

class plot:
    def __init__(self, execute):
        self.epoch = 0
        self.test = False
        self.execute = execute
    
    def saveimg(self, model, ans, idx):
        conv, rnn = model.c.detach()[0], model.r.detach()[0]
        fig = plt.figure(figsize=(12.8, 4.8))
        color = ['tab:blue', 'tab:orange', 'tab:green']
        name = ['LSTM', 'GRU', 'RNN']
        value = (conv.max(1).values-conv.min(1).values).argsort()
        fig.suptitle(f'{ans[0]}')
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        ax1.set_title('conv')
        ax2.set_title('rnn')
        for i, (c, r) in enumerate(zip(conv, rnn)):
            ax1.plot(list(map(float, c)), c=color[i])
            ax2.plot(list(map(float, r)), c=color[value[i]], label=name[i])
        ax2.legend(bbox_to_anchor=(1, 1), frameon=False)
        s = 'test' if self.test else 'epoch_'+str(self.epoch)
        fig.savefig(f'img/{s}/estimate_{idx}')