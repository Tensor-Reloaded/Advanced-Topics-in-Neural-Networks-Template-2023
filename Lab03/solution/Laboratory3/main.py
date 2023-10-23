import torch
import numpy as np
from torch import Tensor
from torchvision.datasets import MNIST


def sigmoid_func(matrix):
    matrix = 1 / (1 + torch.exp(-matrix))
    return matrix


def sigmoid_derivation(matrix):
    return torch.mul(matrix, 1 - matrix)


def forward_propagation(X, first_weights, second_weights, first_bias, second_bias):
    first_weighted_sum = torch.add(torch.matmul(X, first_weights), first_bias)
    first_activation = sigmoid_func(first_weighted_sum)

    second_weighted_sum = torch.add(torch.matmul(first_activation, second_weights), second_bias)
    # return the activation on the hidden layer and the output of the network, which is sigmoid of the second weighted sum
    return first_activation, sigmoid_func(second_weighted_sum)


def back_propagation(X, first_weights, second_weights, first_bias, second_bias, y_true, mu, first_activation, output):
    # First step: Compute the gradient of the error with respect to the weights
    # and the gradient of the error with respect to the bias between hidden and output layer.

    # ∂E/∂W(l) = A(l-1) * sigmoid'(Z(l)) * ∂E/∂A(l)
    # Because in this case we are on last layer (l=L), ∂E/∂A(l) = output - y_true
    gradient_of_error_to_activation_L = torch.subtract(output, y_true)
    derived_sigmoid = sigmoid_derivation(output)

    # With the two values computed above, we compute:
    # - the gradient of the error with respect to the weights
    gradient_of_error_to_weights = torch.transpose(first_activation, 0, 1) @ torch.mul(derived_sigmoid,
                                                                                       gradient_of_error_to_activation_L)

    # - the gradient of the error with respect to the bias
    n = derived_sigmoid.shape[0]
    m = derived_sigmoid.shape[1]
    multiplication_reshaped = torch.reshape(
        torch.flatten(torch.mul(derived_sigmoid, gradient_of_error_to_activation_L)),
        (10, int(m * n / 10)))
    gradient_of_error_to_bias = multiplication_reshaped.mean(axis=1)

    # Now we can update the second weights and bias using the gradients
    second_weights -= mu * gradient_of_error_to_weights
    second_bias -= mu * gradient_of_error_to_bias

    # Second step: Compute the gradient of the error with respect to the weights
    # and the gradient of the error with respect to the bias between input and hidden layer.
    # Because we ar not on the last layer anymore, the value of ∂E/∂A(l) it's a sum that we will compute separately
    gradient_of_error_to_activation_l = second_weights @ torch.transpose(sigmoid_derivation(output), 0,
                                                                         1) @ gradient_of_error_to_activation_L
    reshaped_gradient_of_error_to_activation_l = torch.zeros(100, 100)
    reshaped_gradient_of_error_to_activation_l[0:100, 0:10] = gradient_of_error_to_activation_l
    derived_sigmoid_l = sigmoid_derivation(first_activation)

    # With the two values computed above, we compute:
    # - the gradient of the error with respect to the weights
    gradient_of_error_to_weights_l = torch.transpose(X, 0,
                                                     1) @ derived_sigmoid_l @ reshaped_gradient_of_error_to_activation_l

    # - the gradient of the error with respect to the bias
    n_l = derived_sigmoid_l.shape[0]
    m_l = gradient_of_error_to_activation_l.shape[1]
    multiplication_reshaped_l = torch.reshape(
        torch.flatten(derived_sigmoid_l @ gradient_of_error_to_activation_l),
        (100, int(m_l * n_l / 100)))
    gradient_of_error_to_bias_l = multiplication_reshaped_l.mean(axis=1)
    # Now we can update the first weights and bias using the gradients
    first_weights -= mu * gradient_of_error_to_weights_l
    first_bias -= mu * gradient_of_error_to_bias_l

    loss = torch.nn.functional.cross_entropy(output, y_true).item()
    return first_weights, second_weights, first_bias, second_bias, loss


def train_batch(X, first_weights, second_weights, first_bias, second_bias, y_true, mu):
    first_activation, output = forward_propagation(X, first_weights, second_weights, first_bias, second_bias)

    first_weights, second_weights, first_bias, second_bias, loss = back_propagation(X, first_weights, second_weights,
                                                                              first_bias, second_bias, y_true, mu,
                                                                              first_activation,
                                                                              output)
    return first_weights, second_weights, first_bias, second_bias, loss


def train_epoch(X, first_weights, second_weights, first_bias, second_bias, y_true, mu, batch_size):
    loss_per_epoch = 0
    for i in range(0, data.shape[0], batch_size):
        x = X[i: i + batch_size]
        y = y_true[i: i + batch_size]
        first_weights, second_weights, first_bias, second_bias, loss = train_batch(x,first_weights, second_weights, first_bias, second_bias, y, mu)
        loss_per_epoch+=loss
    return first_weights, second_weights, first_bias, second_bias, loss_per_epoch


def train_network_v2(X, first_weights, second_weights, first_bias, second_bias, y_true, mu):
    nr_epochs = 130
    for epoch in range(nr_epochs):
        first_weights, second_weights, first_bias, second_bias, loss_per_epoch = train_epoch(X, first_weights, second_weights, first_bias, second_bias, y_true, mu, 150)
        print("For epoch {0} the loss equals: {1}".format(epoch+1, loss_per_epoch))
    return first_weights, second_weights, first_bias, second_bias


def test_network(X, first_weights, second_weights, first_bias, second_bias, y_true):
    first_activation, output = forward_propagation(X, first_weights, second_weights, first_bias, second_bias)
    total_test = y_true.size(dim=0)
    correct_prediction = 0
    predicted_label = torch.argmax(output, dim=1)
    for i in range(total_test):
        if y_true[i] == predicted_label[i]:
            correct_prediction += 1
    return correct_prediction / total_test


def collate(x) -> Tensor:
    if isinstance(x, (tuple, list)):
        if isinstance(x[0], Tensor):
            return torch.stack(x)
        return torch.tensor(x)
    raise "Not supported yet"


def to_one_hot(x: Tensor) -> Tensor:
    return torch.eye(x.max() + 1)[x]


def load_mnist(path: str = "./data", train: bool = True):
    mnist_raw = MNIST(path, download=True, train=train)
    mnist_data = []
    mnist_labels = []
    for image, label in mnist_raw:
        tensor = torch.from_numpy(np.array(image))
        mnist_data.append(tensor)
        mnist_labels.append(label)

    mnist_data = collate(mnist_data).float()  # shape 60000, 28, 28
    mnist_data = mnist_data.flatten(start_dim=1)  # shape 60000, 784
    mnist_data /= mnist_data.max()
    mnist_data -= 0.1607
    mnist_data /= 0.4081
    mnist_labels = collate(mnist_labels)  # shape 60000
    if train:
        mnist_labels = to_one_hot(mnist_labels)  # shape 60000, 10
    return mnist_data, mnist_labels


if __name__ == '__main__':
    first_weights = torch.rand(784, 100)
    second_weights = torch.rand(100, 10)

    first_bias = torch.rand(100)
    second_bias = torch.rand(10)

    mu = 0.05
    loss = 0

    data, labels = load_mnist(train=True)
    data_test, labels_test = load_mnist(train=False)

    first_weights, second_weights, first_bias, second_bias = train_network_v2(data, first_weights, second_weights, first_bias, second_bias, labels, mu)

    accuracy = test_network(data_test, first_weights, second_weights, first_bias, second_bias, labels_test)
    print("Accuracy after training: {0}".format(accuracy))