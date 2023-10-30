import matplotlib.pyplot as plt

def draw_chart(chart_name: str, x, y):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(x, y)
    ax.set(title=chart_name, xlabel='Epochs', ylabel='Loss mean')

    fig.savefig(f'{chart_name}.png', format='png')
    plt.show()