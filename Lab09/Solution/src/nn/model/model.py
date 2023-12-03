import torch
from torch.nn import Module, Conv2d, MaxPool2d, Linear, init, ReLU, Identity


class Model(Module):
    __device: torch.device

    def __init__(
        self, device: torch.device = torch.device("cpu"), *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.__device = device

        self.conv1 = Conv2d(3, 32, kernel_size=3)
        self.conv2 = Conv2d(32, 64, kernel_size=3)
        self.pool1 = MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = Linear(64 * 36, 512)
        self.fc2 = Linear(512, 28 * 28)

        init.kaiming_uniform_(self.fc1.weight)
        init.kaiming_uniform_(self.fc2.weight)

        self.activation_function = ReLU()
        self.output_layer_activation_function = Identity()
        self.to(device=self.__device, non_blocking=self.__device == "cuda")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(device=self.__device, non_blocking=self.__device == "cuda")

        x = self.pool1(self.activation_function(self.conv1(x)))
        x = self.pool1(self.activation_function(self.conv2(x)))

        x = x.view(x.shape[0], -1)

        x = self.activation_function(self.fc1(x))
        x = self.activation_function(self.fc2(x))

        x = x.view(-1, 1, 28, 28)

        return x
