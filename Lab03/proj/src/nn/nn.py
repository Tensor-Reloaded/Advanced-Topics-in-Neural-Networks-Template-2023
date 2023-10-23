import torch
import typing as t
from .exceptions import MultilayeredNeuralNetworkException


class MultilayeredNeuralNetwork:
    _layers: t.List[int]
    _learning_rate: float
    _activation_function: t.Callable[[torch.Tensor], torch.Tensor]
    _activation_function_derivative: t.Callable[[torch.Tensor], torch.Tensor]
    _cost_function: t.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    _cost_function_derivative: t.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

    _W: t.List[torch.Tensor]
    _b: t.List[torch.Tensor]

    _device: torch.device

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
        device: torch.device = torch.device("cpu"),
    ) -> None:
        if len(layers) < 2 or len(list(filter(lambda x: x < 0, layers))) > 0:
            raise MultilayeredNeuralNetworkException(
                f"Invalid number of layers: Given {len(layers)}, expected >= 2"
            )

        self._layers = layers
        self._learning_rate = learning_rate
        self._activation_function = activation_function
        self._activation_function_derivative = activation_function_derivative
        self._cost_function = cost_function
        self._cost_function_derivative = cost_function_derivative
        self._device = device

        self._W = [
            torch.randn(inputs, outputs).to(device=device)
            for inputs, outputs in zip(layers[:-1], layers[1:])
        ]
        self._b = [torch.randn(outputs, 1).to(device=device) for outputs in layers[1:]]

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        return self._forward_propagate(X)

    def train_one(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self._backward_propagate(X, y)

    def train_batched(
        self, X: torch.Tensor, y: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        if batch_size < 1:
            raise MultilayeredNeuralNetworkException(
                f"Invalid batch size: Given {batch_size}, expected >= 1"
            )

        batch_loss = 0

        for batch_index in range(0, X.shape[0] // batch_size):
            X_batch = X[batch_size * batch_index : batch_size * (batch_index + 1)].t()
            y_batch = y[batch_size * batch_index : batch_size * (batch_index + 1)].t()

            batch_loss += self.train_one(X_batch, y_batch)

        return batch_loss

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
    ) -> torch.Tensor:
        layers_count = len(self._layers)
        activations = [X]
        weighted_sums = []

        for W, b in zip(self._W, self._b):
            z = W.t() @ activations[-1] + b
            a = self._activation_function(z)

            activations.append(a)
            weighted_sums.append(z)

        delta = self._cost_function_derivative(y, activations[-1]) / X.shape[0]

        for layer_index in range(1, layers_count - 1):
            dW = delta @ activations[-(layer_index + 1)].t()
            db = delta.sum(dim=1, keepdim=True)

            self._W[-layer_index] -= self._learning_rate * dW.t()
            self._b[-layer_index] -= self._learning_rate * db

            delta = self._activation_function_derivative(
                weighted_sums[-(layer_index + 1)]
            ) * (self._W[-layer_index] @ delta)

        loss = self._cost_function(y, activations[-1])

        return loss

    @property
    def learning_rate(self) -> float:
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value: float):
        self._learning_rate = value
