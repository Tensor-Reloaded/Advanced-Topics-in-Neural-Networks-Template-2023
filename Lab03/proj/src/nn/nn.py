import torch
import typing as t
from .exceptions import MultilayeredNeuralNetworkException


class MultilayeredNeuralNetwork:
    _learning_rate: float
    _activation_function: t.Callable[[torch.Tensor], torch.Tensor]
    _activation_function_derivative: t.Callable[[torch.Tensor], torch.Tensor]
    _cost_function: t.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    _cost_function_derivative: t.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

    _W: t.List[torch.Tensor]
    _b: t.List[torch.Tensor]

    def __init__(
        self,
        layers: t.List[int],
        learning_rate: float,
        activation_function: t.Callable[[torch.Tensor], torch.Tensor],
        activation_function_derivative: t.Callable[[torch.Tensor], torch.Tensor],
        cost_function: t.Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        cost_function_derivative: t.Callable[
            [torch.Tensor, torch.Tensor], torch.Tensor
        ],
    ) -> None:
        if len(layers) < 2:
            raise MultilayeredNeuralNetworkException(
                f"Invalid number of layers: Given {len(layers)}, expected >= 2"
            )

        self._learning_rate = learning_rate
        self._activation_function = activation_function
        self._activation_function_derivative = activation_function_derivative
        self._cost_function = cost_function
        self._cost_function_derivative = cost_function_derivative

        self._W = [
            torch.randn(inputs, outputs)
            for inputs, outputs in zip(layers[:-1], layers[1:])
        ]
        self._b = [torch.randn(outputs, 1) for outputs in layers[1:]]

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        return self._forward_propagate(X)

    def train(self, X: torch.Tensor, y: torch.Tensor) -> None:
        self._backward_propagate(X, y)

    def _forward_propagate(self, X: torch.Tensor) -> torch.Tensor:
        a = X

        for W, b in zip(self._W, self._b):
            z = W.t() @ a + b
            a = self._activation_function(z)

        return a

    def _backward_propagate(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
    ) -> None:
        activations = [X]
        weighted_sums = []

        for W, b in zip(self._W, self._b):
            z = W.t() @ activations[-1] + b
            a = self._activation_function(z)

            activations.append(a)
            weighted_sums.append(z)

        delta = self._cost_function_derivative(
            activations[-1], y
        ) * self._activation_function_derivative(weighted_sums[-1])

        for layer_index in range(len(self._W) - 1, -1, -1):
            dW = delta.t() @ activations[layer_index - 1]
            db = torch.sum(delta, dim=1, keepdim=True)

            delta = (
                self._W[layer_index]
                @ delta
                * self._activation_function_derivative(
                    weighted_sums[layer_index - 1]
                ).t()
            )

            self._W[layer_index] -= self._learning_rate * dW
            self._b[layer_index] -= self._learning_rate * db
