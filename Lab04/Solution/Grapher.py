import matplotlib.pyplot as plt


class Grapher:
    @staticmethod
    def draw(name, x, y):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.plot(x, y)
        ax.set(title=name, xlabel='Epochs', ylabel='Loss mean')
        fig.savefig(f'./Plots/{name}.pdf', format='pdf')
        plt.show()
