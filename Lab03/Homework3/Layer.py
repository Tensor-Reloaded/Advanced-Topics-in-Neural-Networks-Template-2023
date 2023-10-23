import math
from typing import Callable

import torch
from torch import Tensor

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




