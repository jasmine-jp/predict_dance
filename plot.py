import matplotlib.pyplot as plt

class plot:
    def __init__(self, execute):
        self.epoch = 0
        self.test = False
        self.execute = execute
    
    def saveimg(self, model, ans, idx):
        fig = plt.figure(figsize=(12.8, 4.8))
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        for x in [model.conv[0][:, i] for i in range(len(model.rnnlist))]:
            ax1.plot(list(map(float, x)))
        for x in model.rnn[0]:
            ax2.plot(list(map(float, x)))
        fig.suptitle(f'{ans[0]}')
        ax1.set_title('conv')
        ax2.set_title('rnn')
        s = 'test' if self.test else 'epoch_'+str(self.epoch)
        fig.savefig(f'img/{s}/estimate_{idx}')