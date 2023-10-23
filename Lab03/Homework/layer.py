import math
import torch
from torch import Tensor
from typing import Optional, Callable


def sigmoid(input_data: Tensor) -> Tensor:
    return 1 / (1 + torch.exp(-input_data))


def softmax(input_data: Tensor) -> Tensor:
    return torch.exp(input_data) / torch.sum(torch.exp(input_data))


ACTIVATION_FNS = {
    "sigmoid": sigmoid,
    "softmax": softmax
}


class LikeLayer:
    def __init__(self, size: int):
        self.size = size
        self.previous: Optional["LikeLayer"] = None


class InputLayer(LikeLayer):
    def __init__(self, size: int):
        super().__init__(size)


class Layer(LikeLayer):
    def __init__(self, size: int, activation: str, previous: "LikeLayer"):
        super().__init__(size)

        self.activation = activation
        self.weights, self.biases = Layer.__initialize_kernel(previous.size, size)
        self.previous = previous

        self.__activation_fn: Callable = None

    @staticmethod
    def __initialize_kernel(previous: int, current: int) -> tuple[Tensor, Tensor]:
        weights = torch.reshape(
            torch.empty(previous * current).normal_(mean=0, std=1 / math.sqrt(previous)),
            (previous, current)
        )

        biases = torch.reshape(
            torch.empty(current).normal_(mean=0, std=1 / math.sqrt(previous)),
            (1, current)
        )

        return weights, biases

    def compile(self) -> list["Layer"]:
        layers = []
        layer_to_insert = self
        while layer_to_insert.previous is not None:
            assert layer_to_insert.activation == "sigmoid" or layer_to_insert.activation == "softmax"
            layer_to_insert.__activation_fn = ACTIVATION_FNS[layer_to_insert.activation]

            layers.insert(0, layer_to_insert)
            layer_to_insert = layer_to_insert.previous
        return layers

    def feed_forward(self, input_data: Tensor) -> Tensor:
        return self.__activation_fn(input_data @ self.weights + self.biases)

    @staticmethod
    def compute_error_output_layer(output: Tensor, target: Tensor) -> Tensor:
        return target - output

    @staticmethod
    def compute_error(output: Tensor, next_error: Tensor, next_layer: "Layer") -> Tensor:
        return next_layer.weights @ next_error

