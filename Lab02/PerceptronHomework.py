import math
import gzip
import pickle

import torch
from torch import Tensor


def forward_pass(X: Tensor, W: Tensor, b: Tensor) -> Tensor:
    z = X @ W + b
    z.apply_(lambda item: 1 / (1 + math.exp(-item))) # y
    return z


def backward_pass(X: Tensor, W: Tensor, b: Tensor, y_true: Tensor, y_pred: Tensor, mu: float) -> tuple[Tensor, Tensor]:
    error = y_true - y_pred
    W += mu * (X.T @ error)
    b += mu * error
    return W, b


def train_perceptron(X: Tensor, W: Tensor, b: Tensor, y_true: Tensor, mu: float) -> tuple[Tensor, Tensor]:
    for it in range(X.size(dim=0)):
        current_x = X[it:it+1, :]
        current_y_true = y_true[it:it+1]

        current_y_pred = forward_pass(current_x, W, b)
        W, b = backward_pass(current_x, W, b, current_y_true, current_y_pred, mu)

    return W, b


def benchmark(X: Tensor, W: Tensor, b: Tensor, y_true: Tensor) -> float:
    correctly_classified = 0

    for it in range(X.size(dim=0)):
        current_x = X[it:it+1, :]
        current_y_true = y_true[it:it+1]

        current_y_pred = forward_pass(current_x, W, b)
        if torch.argmax(current_y_true) == torch.argmax(current_y_pred):
            correctly_classified += 1

    return correctly_classified / X.size(dim=0)


def get_initial_weights_biases() -> tuple[Tensor, Tensor]:
    input_neurons = 784
    output_neurons = 10

    weights = torch.reshape(
        torch.empty(input_neurons * output_neurons).normal_(mean=0, std=1 / math.sqrt(input_neurons)),
        (input_neurons, output_neurons)
    )

    biases = torch.reshape(
        torch.empty(output_neurons).normal_(mean=0, std=1 / math.sqrt(input_neurons)),
        (1, output_neurons)
    )

    return weights, biases


def input_convert(data: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
    features, labels = torch.from_numpy(data[0]), torch.from_numpy(data[1])

    repeat_labels = labels.view(-1, 1).repeat(1, 10)
    repeat_range = torch.arange(0, 10, 1).view(1, -1).repeat(labels.size(dim=0), 1)

    return features, (repeat_labels == repeat_range) * 1.0


def mnist_test():
    weights, biases = get_initial_weights_biases()
    with gzip.open("./data/mnist.pkl.gz") as fd:
        training_set, validation_set, test_set = pickle.load(fd, encoding="latin")

        training_set_features, training_set_labels = input_convert(training_set)
        validation_set_features, validation_set_labels = input_convert(validation_set)
        test_set_features, test_set_labels = input_convert(test_set)

        accuracy = benchmark(test_set_features, weights, biases, test_set_labels)
        print(accuracy)
        # accuracy 11.56%

        weights, biases = train_perceptron(training_set_features, weights, biases,
                                           training_set_labels, 0.001)

        accuracy = benchmark(test_set_features, weights, biases, test_set_labels)
        print(accuracy)
        # accuracy 89.16%


if __name__ == "__main__":
    mnist_test()
