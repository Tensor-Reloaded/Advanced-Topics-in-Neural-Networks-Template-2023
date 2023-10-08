import math
import typing as t
import numpy as np
import numpy.typing as npt

DType = t.TypeVar("DType", bound=np.generic)

Vector = t.Annotated[npt.NDArray[DType], t.Literal["N"]]
Matrix = t.Annotated[npt.NDArray[DType], t.Literal["N", "N", "N"]]


class NeuralNetwork:
    weights: Matrix[np.float32]
    bias: Vector[np.float32]
    learning_rate: np.float32

    def __init__(
        self,
        weights: Matrix[np.float32],
        bias: Vector[np.float32],
        learning_rate: np.float32
    ):
        self.weights = weights
        self.bias = bias
        self.learning_rate = learning_rate

    def predict(self, x: Vector[np.float32]) -> np.float32:
        return self._run_forward_propagation(x)

    def train(self, x: Vector[np.float32], y: np.float32) -> None:
        return self._run_backwards_propagation(x, y)

    def display(self):
        print(f"Weights:\n{self.weights}")
        print(f"Bias:\t\t{self.bias}")
        print(f"Learning rate:\t{self.learning_rate}")

    def _run_forward_propagation(self, x: Vector[np.float32]) -> np.float32:
        wx_plus_b = np.dot(self.weights.T, x) + self.bias
        y_prime = self._run_activation_function(wx_plus_b)

        return y_prime

    def _run_activation_function(self, z: Vector[np.float32]) -> Vector[np.float32]:
        exp_z = np.exp(z - np.max(z))
        return exp_z / np.sum(exp_z)

    def _run_backwards_propagation(self, x: Vector[np.float32], y: np.float32):
        y_prime = self._run_forward_propagation(x)

        error = self._run_loss_function(y, y_prime)

        delta_weights = np.dot(error, x.T)
        delta_biases = error

        self.weights = self.weights - self.learning_rate * delta_weights
        self.bias = self.bias - self.learning_rate * delta_biases

    def _run_cross_entropy_loss_function(self, y: np.float32, y_prime: np.float32) -> np.float32:
        return -(y * math.log(y_prime) + (1 - y) * math.log(1 - y_prime))

    def _run_loss_function(self, y: Vector[np.float32], y_prime: Vector[np.float32]) -> np.float32:
        return y_prime - y
