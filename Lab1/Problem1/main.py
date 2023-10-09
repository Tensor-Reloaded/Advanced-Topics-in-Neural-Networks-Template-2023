import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


def cross_entropy(output, target):
    return -np.sum(target * np.log(output))


def cross_entropy_derivative(output, target):
    return output - target


if __name__ == "__main__":
    lr = 0.02

    x = np.array([1, 3, 0])
    target = np.array([0, 1, 0])

    weights = np.array([[0.3, 0.1, -2], [-0.6, 0.5, 2], [-1, -0.5, -0.1]])
    biases = np.array([0.1, 0.1, 0.1])

    y = np.dot(weights, x) + biases
    z = softmax(y)

    loss = cross_entropy(z, target)
    print("Loss: ", loss)

    loss_derivative = cross_entropy_derivative(z, target)
    gradient_weights = np.dot(loss_derivative, x)
    gradient_biases = loss_derivative

    weights -= lr * gradient_weights
    biases -= lr * gradient_biases

    print("Weights: ", weights)
    print("Biases: ", biases)
