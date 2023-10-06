#!/usr/bin/env python3
import numpy as np
import math
import numpy.typing as npt

Tensor = npt.NDArray[np.float32]

def main():
    x = np.array([1, 3, 0], dtype=np.float32)
    w = np.array([-0.6, -0.5, 2], dtype=np.float32)
    b = np.array([0.1, 0.1, 0.1], dtype=np.float32)
    y = np.float32(1)

    neural_network = NeuralNetwork(w, b)
    w_prime, b_prime = neural_network.iterate(x, y)

    print(w_prime)
    print(b_prime)

class NeuralNetwork:
    weights: Tensor
    bias: np.float32

    def __init__(self, weights: Tensor, bias: np.float32):
        self.weights = weights
        self.bias = bias

    def forward_propagation(self, x: Tensor) -> np.float32:
        wx_plus_b = np.sum(self.weights * x) + np.sum(self.bias)
        y_prime = self.activation_function(wx_plus_b)

        return y_prime

    def iterate(self, x: Tensor, y: np.float32):
        y_prime = self.forward_propagation(x)

        error = self.loss_function(y, y_prime)

        delta_weights = -error * self.weights
        delta_biases = -error * np.ones_like(self.bias)

        weights_prime = self.weights - delta_weights
        biases_prime = self.bias - delta_biases

        return weights_prime, biases_prime

    def activation_function(self, x: np.float32) -> np.float32:
        return 1 / (1 + math.e ** -x)

    def loss_function(self, y: np.float32, y_prime: np.float32) -> np.float32:
        return y - y_prime

if __name__ == '__main__':
    main()
