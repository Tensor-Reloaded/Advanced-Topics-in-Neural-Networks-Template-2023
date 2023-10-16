import _pickle
import gzip

import torch
from torch import Tensor


def get_dataset():
    f = gzip.open('mnist.pkl.gz', 'rb')
    train, valid, test = _pickle.load(f, encoding="latin1")
    f.close()
    return train, test


def get_prediction_accuracy(instances, labels, weights, biases):
    good_results = 0
    output = torch.zeros(10)
    for iterator in range(10000):
        x = torch.tensor(instances[iterator])
        t = labels[iterator]
        for digit in range(10):
            output[digit] = torch.dot(x, weights[:, digit]) + biases[digit]
        if output.argmax() == t:
            good_results += 1
    print(good_results / 100, "%")


def train_perceptron(X: Tensor, W: Tensor, b: Tensor, y_true: Tensor, mu: float):
    n_instances = X.shape[0]
    n_classes = b.shape[0]

    for c in range(n_classes):
        W_column = W[:, c]
        n_iter = 3
        while n_iter > 0:
            for i in range(n_instances):
                z = torch.dot(W_column, X[i]) + b[c]
                sigmoid_z = 1 / (1 + torch.exp(-z))
                Error = y_true[i, c] - sigmoid_z
                W_column += mu * Error * X[i]
                b[c] += mu * Error
            print(f'iteration {n_iter} completed')
            n_iter -= 1

        W[:, c] = W_column

    return W, b


def process_labels(labels):
    new_labels = torch.zeros((len(labels), 10))
    for label in range(len(labels)):
        new_labels[label][labels[label]] = 1

    return new_labels


if __name__ == '__main__':
    train_set, test_set = get_dataset()

    # input_features = torch.rand((10000, 784))
    initial_weights = torch.rand((784, 10))
    initial_biases = torch.rand(10)
    # true_labels = torch.ones((10000, 10))

    input_features, true_labels = train_set[0], train_set[1]
    encoded_true_labels = process_labels(true_labels)

    updated_weights, updated_biases = train_perceptron(torch.tensor(input_features), initial_weights, initial_biases,
                                                       encoded_true_labels, 0.1)

    test_instances, test_labels = test_set[0], test_set[1]
    get_prediction_accuracy(test_instances, test_labels, updated_weights, updated_biases)
