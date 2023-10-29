import typing as t
import matplotlib.pyplot as plt

def graph(name: str, data: t.List[t.Tuple[int, float]]):
    x, y = zip(*data)

    plt.figure(figsize=(10,6))
    plt.plot(x, y)

    plt.title(name)
    plt.xlabel('Epochs')
    plt.ylabel('Loss mean')

    plt.savefig(f"{name}.pdf", format="pdf")
    plt.show()
