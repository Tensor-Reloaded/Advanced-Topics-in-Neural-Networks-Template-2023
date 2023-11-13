from torch import nn, Tensor


class ZAE(nn.Module):
    def __init__(self, treshold: float = 1.0):
        super(ZAE, self).__init__()
        self.treshold = treshold

    def forward(self, input: Tensor) -> Tensor:
        mask = input <= self.treshold
        input[mask] = 0
        return input

