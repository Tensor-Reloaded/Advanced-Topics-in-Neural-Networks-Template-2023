import math
import torch
from torch import Tensor
from typing import Optional, Callable


def sigmoid(input_data: Tensor) -> Tensor:
    return 1 / (1 + torch.exp(-input_data))


def softmax(input_data: Tensor) -> Tensor:
    return torch.exp(input_data) / torch.sum(torch.exp(input_data), dim=1).view(-1, 1)


ACTIVATION_FNS = {
    "sigmoid": sigmoid,
    "softmax": softmax
}

HIDDEN_LAYER_ERROR = {
    "sigmoid": lambda output, next_error, next_layer: output * (1 - output) * (next_error @ next_layer.weights.T)
}

OUTPUT_LAYER_ERROR = {
    ("softmax", "cross_entropy"): lambda output, target: target - output
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
            layer_to_insert.__activation_fn = ACTIVATION_FNS[layer_to_insert.activation]

            layers.insert(0, layer_to_insert)
            layer_to_insert = layer_to_insert.previous
        return layers

    def feed_forward(self, input_data: Tensor) -> Tensor:
        res = self.__activation_fn(input_data @ self.weights + self.biases)
        assert torch.any(torch.isnan(res)) == False
        return res

    @staticmethod
    def compute_error_output_layer(output: Tensor, target: Tensor) -> Tensor:
        return target - output

    @staticmethod
    def compute_error(output: Tensor, next_error: Tensor, next_layer: "Layer") -> Tensor:
        return output * (1 - output) * (next_error @ next_layer.weights.T)

