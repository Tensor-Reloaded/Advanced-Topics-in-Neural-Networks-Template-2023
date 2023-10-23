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

HIDDEN_LAYER_ERROR_FNS = {
    "sigmoid": lambda output, next_error, next_layer: output * (1 - output) * (next_error @ next_layer.weights.T)
}

OUTPUT_LAYER_ERROR_FNS = {
    ("softmax", "cross_entropy"): lambda output, target: output - target
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
        self.__hidden_layer_error_fn: Callable = None
        self.__output_layer_error_fn: Callable = None

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

        return weights.double(), biases.double()

    def compile(self, loss: str) -> list["Layer"]:
        layers = []
        layer_to_insert = self
        while layer_to_insert.previous is not None:
            layer_to_insert.__activation_fn = ACTIVATION_FNS[layer_to_insert.activation]

            if len(layers) == 0:
                layer_to_insert.__output_layer_error_fn = OUTPUT_LAYER_ERROR_FNS[(layer_to_insert.activation, loss)]
            else:
                layer_to_insert.__hidden_layer_error_fn = HIDDEN_LAYER_ERROR_FNS[layer_to_insert.activation]

            layers.insert(0, layer_to_insert)
            layer_to_insert = layer_to_insert.previous
        return layers

    def feed_forward(self, input_data: Tensor) -> Tensor:
        res = self.__activation_fn(input_data @ self.weights + self.biases)
        assert not torch.any(torch.isnan(res))
        return res

    def compute_error_output_layer(self, output: Tensor, target: Tensor) -> Tensor:
        return self.__output_layer_error_fn(output, target)

    def compute_error(self, output: Tensor, next_error: Tensor, next_layer: "Layer") -> Tensor:
        return self.__hidden_layer_error_fn(output, next_error, next_layer)

