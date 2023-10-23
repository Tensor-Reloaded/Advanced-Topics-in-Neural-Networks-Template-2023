import torch
from torch import Tensor
from typing import Callable

import util
from Layer import Layer


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
        print(f"Accuracy: {correct / total}")



