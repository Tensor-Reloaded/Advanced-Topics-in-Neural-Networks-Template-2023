import torch
import nn

def exercise_1():
    m = 100
    X = torch.rand(m, 784)
    W = torch.rand(784, 10)
    b = torch.rand(10, 1)
    y_true = torch.rand(m, 10)
    mu = torch.rand(1)

    trained_W, trained_b = train_perceptron(X, W, b, y_true, mu)

    print("Trained weights:")
    print(trained_W)
    print()
    print("Trained biases:")
    print(trained_b)


def train_perceptron(
    X: torch.Tensor, W: torch.Tensor, b: torch.Tensor, y_true: torch.Tensor, mu: float
):
    current_W = W
    current_b = b

    for entry, labels in zip(X, y_true):
        current_W, current_b = nn.backward_propagation(
            entry.view(-1, 1), current_W, current_b, labels.view(-1, 1), mu
        )

    return current_W, current_b

