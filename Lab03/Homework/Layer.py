import math

import torch


class Layer:
    def __init__(self, input_size, output_size, activation_callback):
        self.input_size = input_size
        self.output_size = output_size
        self.input = None
        self.output = None
        self.activated_output = None
        self.activation = activation_callback
        self.weights = torch.randn(output_size, input_size) * math.sqrt(1 / input_size)
        self.biases = torch.randn(output_size, 1)

    def feedforward(self, input_values):
        self.input = input_values
        self.output = torch.matmul(self.weights, input_values) + self.biases
        self.activated_output = self.activation(self.output)
