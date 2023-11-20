import torch


class Convolution:
    __weights: torch.Tensor

    def __init__(self, weights: torch.Tensor) -> None:
        self.__weights = weights

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs
