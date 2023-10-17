import gzip
import pickle

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import Tensor

with gzip.open(r'data/mnist.pkl.gz', 'rb') as fd:
    train_set, valid_set, test_set = pickle.load(fd, encoding='latin')


# def get_normal_tensors(x: Tensor) -> Union[Tensor, None]:
#     mean = torch.mean(x)
#     norms = torch.norm(x, dim=(1, 2))
#     stdev = torch.std(norms)
#     mask1 = -1.5 * stdev + mean < norms
#     mask2 = 1.5 * stdev + mean > norms
#     mask = mask1 & mask2
#     print(mean)
#     print(norms)
#     print(stdev)
#     print(mask)
#     print(torch.cuda.is_available())

def sigmoid_activation(z: Tensor):
    return 1 / (1 + torch.exp(-z))


def shuffle_in_unison(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)


#   X(m, 784); W(784, 10); b(10,); y_true(m, 10)
#   For online training, m=1 will be used
def train_perceptron(X: Tensor, W: Tensor, b: Tensor, y_true: Tensor, mu: float):
    #   Weighted Sum (m, 10)
    z = X @ W + b

    #   Activation (m, 10)
    a = sigmoid_activation(z)

    #   Compute error (m, 10)
    error = a - y_true

    #   Compute the change to W: (784, m) x (m, 10) = (784, 10)
    delta_W = torch.transpose(X, 0, 1) @ error * mu

    #   Compute the change to b: sum(m, 10) = (10,)
    delta_b = torch.sum(error, dim=0)

    return W - delta_W, b - delta_b


def mini_batch_training(epochs: int, batch_size: int, mu: float, training_set: Tensor, input_size: int,
                        output_size: int, W: Tensor, b: Tensor):
    for iteration in range(epochs):
        batch_start = 0
        while True:
            curr_batch_size = min(batch_size, len(training_set[0]) - batch_start)
            X = torch.tensor(training_set[0][batch_start: batch_start + curr_batch_size])
            # transition from number to vector where vector[number] = 1, otherwise 0
            y_true_nr = training_set[1][batch_start: batch_start + curr_batch_size]
            y_true = torch.zeros((curr_batch_size, output_size))
            for i in range(curr_batch_size):
                y_true[i, y_true_nr[i]] = 1

            W, b = train_perceptron(X, W, b, y_true, mu)

            batch_start += curr_batch_size
            if batch_start >= len(training_set[0]):
                break
        shuffle_in_unison(training_set[0], training_set[1])

    return W, b


def test_instances(testing_set: (Tensor, Tensor), weights: Tensor, biases: Tensor):
    accurate = 0
    inaccurate = 0
    for index in range(len(testing_set[0])):
        coordinates = testing_set[0][index]
        actual_value = testing_set[1][index]
        prediction = get_prediction(coordinates, weights, biases)
        if actual_value == prediction:
            accurate += 1
        else:
            inaccurate += 1
    return accurate, inaccurate


def get_prediction(test_case, weights, biases):
    z = (test_case @ weights) + biases
    prediction = sigmoid_activation(z)
    return torch.argmax(prediction)


if __name__ == '__main__':
    # m = 2
    # X = torch.rand((m, 784))
    # W = torch.rand((784, 10))
    # b = torch.rand((10, ))
    # y_true = (torch.rand((m, 10)) > 0.5).int()
    # mu = 0.2
    #
    # W_trained, b_trained = train_perceptron(X, W, b, y_true, mu)
    # print("W_trained.shape = ", W_trained.shape)
    # print("W_trained = \n", W_trained)
    # print("b_trained.shape = ", b_trained.shape)
    # print("b_trained = \n", b_trained)

    input_size = 784
    output_size = 10
    mu = 0.2
    W = torch.rand((input_size, output_size))
    b = torch.rand((output_size,))

    print("Results on test set with no training: ",
          test_instances((torch.tensor(test_set[0]), torch.tensor(test_set[1])), W, b), " (accurate, inaccurate)")

    epochs = 1
    batch_size = 1  # online training
    W, b = mini_batch_training(1, 1, mu, train_set, input_size, output_size, W, b)

    training_test_set = torch.tensor(train_set[0][:10000]), torch.tensor(train_set[1][:10000])
    print("Test on training data after 1 iteration of training:", test_instances(training_test_set, W, b),
          " (accurate, inaccurate)")
    print("Results on test set after 1 iteration of training: ",
          test_instances((torch.tensor(test_set[0]), torch.tensor(test_set[1])), W, b), " (accurate, inaccurate)")

    # Results on test set with no training:  (998, 9002)  (accurate, inaccurate)
    # Test on training data : (8673, 1327)  (accurate, inaccurate)
    # Results on test set after 1 iteration of training:  (8785, 1215)  (accurate, inaccurate)
