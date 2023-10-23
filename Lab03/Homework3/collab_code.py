import math
from typing import Callable

import torch
from torch import Tensor
from torchvision import datasets
from functools import wraps, partial

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(torch.cuda.get_device_name(0))

class Layer:
    def __init__(self, input_count: int, output_count: int):
        self.error = None
        self.activation = None
        self.input_count = input_count
        self.output_count = output_count
        std_dev = 1.0 / math.sqrt(input_count)
        self.weights = torch.randn((input_count, output_count)) * std_dev
        self.biases = torch.rand(1, output_count)

    def eval(self, data: Tensor) -> Tensor:
        z = (data @ self.weights) + self.biases
        return z

    def calc_activation(self, output: Tensor, activation_function: Callable[[Tensor], Tensor]):
        activation = []
        for i in range(output.shape[0]):
            activation.append(activation_function(output[i]))
        activation = torch.stack(activation)
        self.activation = activation
        return activation





class MLP:
    def __init__(self, layer_counts):
        self.layers = []
        for i in range(len(layer_counts) - 1):
            self.layers.append(Layer(layer_counts[i], layer_counts[i + 1]))

    def forward_prop(self, data):
        input_data = data
        for layer in self.layers[:-1]:
            output = layer.eval(input_data)
            layer.calc_activation(output, util.sigmoid)
            input_data = layer.activation
        output_layer = self.layers[-1]
        output = output_layer.eval(input_data)
        output_layer.calc_activation(output, util.softmax)
        return output_layer.activation

    def backprop(self, train_set, train_labels, learning_rate):
        output_layer = self.layers[-1]
        hidden_layer = self.layers[-2]
        # forward pass
        output = self.forward_prop(train_set)
        # backward pass
        output_layer.error = output - train_labels

        output_layer_grad_w = hidden_layer.activation.t() @ output_layer.error
        output_layer_grad_b = output_layer.error

        hidden_layer.error = hidden_layer.activation * (1 - hidden_layer.activation) * (
                output_layer.error @ output_layer.weights.t())

        # cahsdasdoqwiejd
        hidden_layer_grad_w = train_set.t() @ hidden_layer.error
        hidden_layer_grad_b = hidden_layer.error
        # sum output_grad_b

        output_layer.weights -= learning_rate * output_layer_grad_w
        output_layer.biases += torch.sum(-learning_rate * output_layer_grad_b, dim=0)

        hidden_layer.weights -= learning_rate * hidden_layer_grad_w
        hidden_layer.biases += torch.sum(-learning_rate * hidden_layer_grad_b, dim=0)

    def shuffle(self, train_set, train_labels):
        indices = torch.randperm(train_set.size(0))

        train_set_shuffled = train_set[indices]
        train_labels_shuffled = train_labels[indices]

        return train_set_shuffled, train_labels_shuffled

    def train(self, train_set, train_labels, batch_size=100, lr=0.01, epochs=100):

        for epoch in range(epochs):
            train_set, train_labels = self.shuffle(train_set, train_labels)

            for i in range(0, train_set.size(0), batch_size):
                batch_set = train_set[i:i + batch_size]
                batch_labels = train_labels[i:i + batch_size]
                self.backprop(batch_set, batch_labels, lr)

            if epoch % 10 == 0:
                print(f"Epoch {epoch} done")
                print(f"Loss: {util.cross_entropy_error(self.forward_prop(train_set), train_labels)}")
                # print("Accuracy: ", self.test(train_set, train_labels))
                print('---------------------------------------')

    def test(self, test_set, test_labels):
        correct = 0
        total = 0
        for i in range(len(test_set)):
            output = self.forward_prop(test_set[i])
            idx_max = torch.argmax(output)
            correct_idx = torch.argmax(test_labels[i])
            if idx_max == correct_idx:
                correct += 1
            total += 1
        return correct / total



class util:
  @partial(torch.jit.trace, example_inputs=(torch.rand((200, 100)).cuda()))
  def sigmoid(z: Tensor) -> Tensor:
      return 1 / (1 + torch.exp(-z))

  def sigmoid_derivative(z: Tensor) -> Tensor:
      return sigmoid(z) * (1 - sigmoid(z))

  def error(y: Tensor, y_true: Tensor) -> Tensor:
      return y - y_true

  @partial(torch.jit.trace, example_inputs=(torch.rand((200, 100)).cuda()))
  def softmax(z: Tensor) -> Tensor:
      return torch.exp(z) / torch.sum(torch.exp(z))

  def softmax_derivative(z: Tensor) -> Tensor:
      return softmax(z) * (1 - softmax(z))

  def cross_entropy_error(y: Tensor, y_true: Tensor) -> Tensor:
      return torch.nn.functional.cross_entropy(y, y_true)

  def convert_to_one_hot(y):
      one_hot = torch.zeros(y.size(0), 10)
      for i in range(y.size(0)):
          one_hot[i][y[i]] = 1
      return one_hot


mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)

x_train = mnist_trainset.data
x_train = torch.flatten(x_train, start_dim=1)
y_train_init = mnist_trainset.targets
y_train = util.convert_to_one_hot(y_train_init)

x_train = x_train.float()
y_train = y_train.float()

x_train = x_train / 255

x_test = mnist_testset.data
x_test = torch.flatten(x_test, start_dim=1)
y_test_init = mnist_testset.targets
y_test = util.convert_to_one_hot(y_test_init)

x_test = x_test.float()
y_test = y_test.float()

x_test = x_test / 255

model = MLP([784, 100, 10])
model.train(x_train, y_train, batch_size=100, lr=0.01, epochs=100)

# Test model
acc = model.test(x_test, y_test)
print(f"Accuracy: {acc}")

# tested on Google colab, it does work in under 5 minutes with >95% accuracy
