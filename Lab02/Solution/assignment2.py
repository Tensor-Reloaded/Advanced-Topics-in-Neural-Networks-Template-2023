import numpy as np
import torch
import torchvision
from torch import Tensor
from typing import Tuple
import matplotlib.pyplot as plt
from torchvision.transforms import transforms


def apply_sigmoid(z: Tensor) -> Tensor:
    return 1.0 / (1 + torch.exp(-z))
    # return torch.sigmoid(z)


def train_perceptron(x: Tensor, w: Tensor, b: Tensor, y_true: Tensor, mu: float,
                     batch_size) -> Tuple[Tensor, Tensor]:
    assert x.shape == (batch_size, 784), "Wrong shape for the input features Tensor"
    assert w.shape == (784, 10), "Wrong shape for the initial weights Tensor"
    assert b.shape == (10,), "Wrong shape for the initial biases Tensor"
    assert y_true.shape == (batch_size, 10), "Wrong shape for the ground truth Tensor"
    assert type(mu) == float, "Learning rate should be a float number"

    z = x @ w + b
    y_hat = apply_sigmoid(z)
    # error = (y_true - y_hat) * (y_hat * (1 - y_hat))
    error = y_true - y_hat

    w += mu * (x.transpose(0, 1) @  error.mean(axis=0).unsqueeze(0).repeat(batch_size, 1))
    # w += mu * (x.transpose(0, 1) @  error)
    b += (mu * error.mean(axis=0))

    assert w.shape == (784, 10), "Wrong shape for the updated weights Tensor"
    assert b.shape == (10,), "Wrong shape for the updated biases Tensor"
    return w, b


def evaluate(x_test: Tensor, y_test: Tensor, w: Tensor, b: Tensor):
    z = x_test @ w + b
    y_hat = apply_sigmoid(z)
    y_predicted = torch.stack([torch.where(instance == max(instance), 1.0, 0.0) for instance in y_hat])
    for index, instance in enumerate(y_predicted):
        if int(sum(instance)) != 1:
            y_predicted[index] = Tensor([0.0] * 10)
    return sum((y_test @ y_predicted.transpose(0, 1)).diagonal())


def load_dataset():
    mnist_trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True,
                                                transform=transforms.Compose
                                                ([transforms.ToTensor(),
                                                  # transforms.Normalize(0.1307, 0.3081)
                                                  ]))
    # 60.000 tuple (in, out)
    mnist_testset = torchvision.datasets.MNIST(root='./data', train=False, download=True,
                                               transform=transforms.Compose
                                               ([transforms.ToTensor(),
                                                 # transforms.Normalize(0.1307, 0.3081)
                                                 ]))
    x_train = mnist_trainset.data.float()
    # for index, observation in enumerate(x_train):
    #     std = observation.std()
    #     x_train[index] -= observation.mean()
    #     x_train[index] /= std
    std = x_train.std() / 255.0
    mean = x_train.mean() / 255.0
    x_train -= mean
    x_train /= std
    y_train = mnist_trainset.targets
    y_train = torch.cat([torch.cat((torch.zeros(size=(1, target)),
                                    torch.ones(size=(1, 1)),
                                    torch.zeros(size=(1, 9 - target))), dim=1) for target in y_train])
    x_test = mnist_testset.data.float()
    x_test -= mean
    x_test /= std
    # for index, observation in enumerate(x_test):
    #     std = observation.std()
    #     x_test[index] -= observation.mean()
    #     x_test[index] /= std
    y_test = mnist_testset.targets
    y_test = torch.cat([torch.cat((torch.zeros(size=(1, target)),
                                   torch.ones(size=(1, 1)),
                                   torch.zeros(size=(1, 9 - target))), dim=1) for target in y_test])
    x_train = x_train.to('cuda')
    y_train = y_train.to('cuda')
    x_test = x_test.to('cuda')
    y_test = y_test.to('cuda')
    return x_train.reshape(-1, 784), y_train, x_test.reshape(-1, 784), y_test


def initialize_weights():
    return torch.rand((784, 10)).to('cuda')


def initialize_biases():
    return torch.rand((10,)).to('cuda')


def train(x_train, y_train, x_test, y_test, mu=0.1, batch_size=1, epochs=1000, nsteps=125):
    weights = initialize_weights()
    biases = initialize_biases()
    accuracy_train = []
    accuracy_test = []
    for epoch in range(epochs):
        for step in range(nsteps):
            #  select batch
            batch = np.random.choice(range(x_train.shape[0]), size=batch_size, replace=False)
            input = x_train[batch]
            output = y_train[batch]
            weights, biases = train_perceptron(input, weights, biases, output, mu, batch_size)
        print(f"Finished epoch {epoch}.")
        if epoch % 10 == 0:
            accuracy_test += [evaluate(x_test, y_test, weights, biases).item() / x_test.shape[0]]
            accuracy_train += [evaluate(x_train, y_train, weights, biases).item() / x_train.shape[0]]
            print(accuracy_train)
            print(accuracy_test)
    draw_plot(accuracy_test, epochs, accuracy_train)


def draw_plot(accuracies: list, epochs: int, t_accuracies: list):
    x_axis = list(range(1, epochs + 1, 10))
    plt.plot(x_axis, accuracies)
    plt.plot(x_axis, t_accuracies)
    plt.legend(['testing', 'training'], loc="lower right")
    plt.title('Evolution of accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()


if __name__ == "__main__":
    train_x, train_y, test_x, test_y = load_dataset()
    train(train_x, train_y, test_x, test_y, mu=0.1, epochs=101, nsteps=7500, batch_size=8)
