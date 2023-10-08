#!/usr/bin/env python3
import numpy as np
import math
import numpy.typing as npt

Tensor = npt.NDArray[np.float32]

def main():
    x = np.array([1, 3, 0], dtype=np.float32)
    w = np.array([-0.6, -0.5, 2], dtype=np.float32)
    l = np.float32(1)
    b = np.float32(0.1)
    y = np.float32(1)

    neural_network = NeuralNetwork(w, b, l)
    neural_network.train(x, y)
    neural_network.print()

class NeuralNetwork:
    weights: Tensor
    bias: np.float32
    learning_rate: np.float32

    def __init__(self, weights: Tensor, bias: np.float32, learning_rate: np.float32):
        self.weights = weights
        self.bias = bias
        self.learning_rate = learning_rate

    def predict(self, x: Tensor) -> np.float32:
        return self._run_forward_propagation(x)

    def train(self, x: Tensor, y: np.float32) -> None:
        return self._run_backwards_propagation(x, y)

    def print(self):
        print(f"Weights:\t{self.weights}")
        print(f"Bias:\t\t{self.bias}")
        print(f"Learning rate:\t{self.learning_rate}")

    def _run_forward_propagation(self, x: Tensor) -> np.float32:
        wx_plus_b = np.sum(self.weights.T * x) + np.sum(self.bias)
        y_prime = self._run_activation_function(wx_plus_b)

        return y_prime

    def _run_activation_function(self, x: np.float32) -> np.float32:
        return 1 / (1 + math.exp(-x))

    def _run_backwards_propagation(self, x: Tensor, y: np.float32):
        y_prime = self._run_forward_propagation(x)

        error = self._run_loss_function(y, y_prime)

        delta_weights = error * x
        delta_biases = error

        self.weights = self.weights - self.learning_rate * delta_weights
        self.bias = self.bias - self.learning_rate * delta_biases

    def _run_cross_entropy_loss_function(self, y: np.float32, y_prime: np.float32) -> np.float32:
        return -(y * math.log(y_prime) + (1 - y) * math.log(1 - y_prime))

    def _run_loss_function(self, y: np.float32, y_prime: np.float32) -> np.float32:
        return y_prime - y

if __name__ == '__main__':
    main()
