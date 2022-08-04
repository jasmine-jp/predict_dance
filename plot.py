import matplotlib.pyplot as plt

class plot:
    def __init__(self, execute):
        self.epoch = 0
        self.test = False
        self.execute = execute
    
    def saveimg(self, model, ans, idx):
        fig = plt.figure()
        for x in [model.conv[0][:, i] for i in range(len(model.rnnlist))]:
            plt.plot(list(map(float, x)))
        plt.ylim(float(model.conv[0].min()), float(model.conv[0].max()))
        fig.suptitle(f'{ans[0]}')
        s = 'test' if self.test else 'epoch_'+str(self.epoch)
        fig.savefig(f'img/{s}/conv_{idx}')