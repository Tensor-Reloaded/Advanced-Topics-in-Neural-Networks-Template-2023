import torch
from torch import Tensor
from sklearn.metrics import accuracy_score
from Lab02.parser import InputParser


def sigmoid(t: Tensor):
    return 1 / (1 + torch.exp(-t))


def get_accuracy(yhat, y_true):
    argmax_yhat = torch.argmax(yhat, dim=1)
    argmax_y = torch.argmax(y_true, dim=1)

    matching_positions = argmax_yhat.eq(argmax_y)

    # Count the number of matching positions
    count = matching_positions.sum().item()
    return count / yhat.shape[0] * 100


def train_perceptron(x: Tensor, w: Tensor, b: Tensor, y_true: Tensor = None, mu: float = None):
    z = (x @ w)
    z = z + b
    yhat = sigmoid(z)

    err = yhat - y_true
    delta_w = x.T @ (err * yhat * (1 - yhat)) / x.shape[0]
    delta_b = torch.mean(err * yhat * (1 - yhat), dim=0)

    w -= mu * delta_w
    b = mu * delta_b

    if mu > 0.1:
        mu -= 0.001

    print(get_accuracy(yhat, y_true))
    return w, b, mu


if __name__ == '__main__':
    parser = InputParser(filename="./archive/mnist_train.csv")
    x, y_true = parser.parse()

    weight_shape = (784, 10)
    w = torch.randn(weight_shape)
    b = torch.randn(10)
    mu = 0.1
    # WIP for extra task
    for epoch_index in range(100_000):
        if epoch_index % 100 == 0:
            print(f"Processing batch with index {epoch_index // 100}")
        w, b, mu = train_perceptron(x, w, b, y_true, mu)

    # a = torch.randn((200, 10))
    # b = torch.randn(10)
    # c = a + b
