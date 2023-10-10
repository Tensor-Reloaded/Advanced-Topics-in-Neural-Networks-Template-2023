import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


def cross_entropy(output, target):
    return -np.sum(target * np.log(output))


def cross_entropy_derivative(output, target):
    return output - target

def train(x, target, initial_weights, initial_biases,
          activation_func, loss_func, loss_derivative_func, lr=0.2):
    weights = initial_weights
    biases = initial_biases
    z = np.dot(weights.T, x) + biases
    y = activation_func(z)

    loss = loss_func(y, target)
    print("Loss: ", loss)

    loss_derivative = loss_derivative_func(y, target)
    gradient_weights = np.dot(loss_derivative.reshape(3, 1), x.reshape(1, 3))
    gradient_biases = loss_derivative

    weights -= lr * gradient_weights
    biases -= lr * gradient_biases

    print("Weights: ", weights)
    print("Biases: ", biases)

    return weights, biases


if __name__ == "__main__":
    x = np.array([1, 3, 0])
    target = np.array([0, 1, 0])

    weights = np.array([[0.3, 0.1, -2], [-0.6, -0.5, 2], [-1, -0.5, 0.1]])
    biases = np.array([0.1, 0.1, 0.1])

    train(x, target, weights, biases, softmax, cross_entropy, cross_entropy_derivative)


# Output:
# Loss:  5.408566554451391
# Weights:  [[ 0.29918962  0.09756885 -2.        ]
#  [-0.40089561  0.09731317  2.        ]
#  [-1.19829401 -1.09488202  0.1       ]]
# Biases:  [ 0.09918962  0.29910439 -0.09829401]

