import torch
from math import sqrt
from typing import Callable

class Layer:
    def __init__(self, input_layer_size: int, output_layer_size: int, activation_function: Callable):
        self.input_size = input_layer_size
        self.output_size = output_layer_size
        self.activation = activation_function

        self.input = None
        self.output = None
        self.activated_output = None

        variance = 1.0 / sqrt(input_layer_size)
        self.weights = torch.FloatTensor(output_layer_size, input_layer_size).uniform_(-variance, variance)
        self.biases = torch.FloatTensor(output_layer_size).normal_(0, 1)

        print(f"Successfully initialized layer:\n  Weights shape: {self.weights.shape}\n  Biases shape: {self.biases.shape}")

    def feedforward(self, input_values):
        self.input = input_values
        self.output = torch.matmul(self.weights, input_values) + self.biases
        self.activated_output = self.activation(self.output)
