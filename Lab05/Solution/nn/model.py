import typing as t
import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    activation_function: t.Callable[[torch.Tensor], torch.Tensor]
    output_layer_activation_function: t.Callable[[torch.Tensor], torch.Tensor]
    device: str

    def __init__(
        self,
        input_size: int,
        output_size: int,
        output_layer_activation_function: t.Union[
            None, t.Callable[[torch.Tensor], torch.Tensor]
        ] = None,
        device: str = "cpu",
    ) -> None:
        super(NeuralNetwork, self).__init__()
        self.device = device

        self.fc1 = nn.Linear(input_size, 256).to(
            device=self.device, non_blocking=self.device == "cuda"
        )
        self.fc2 = nn.Linear(256, 128).to(
            device=self.device, non_blocking=self.device == "cuda"
        )
        self.fc3 = nn.Linear(128, 64).to(
            device=self.device, non_blocking=self.device == "cuda"
        )
        self.fc4 = nn.Linear(64, output_size)

        self.activation_function = nn.LeakyReLU()
        self.output_layer_activation_function = (
            output_layer_activation_function
            if output_layer_activation_function is not None
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(device=self.device, non_blocking=self.device == "cuda")

        # FIXME: Before call, tensor is of shape [batch_size, input_size]. Here it is [input_size]...
        if len(x.shape) == 1:
            x = x.view(1, -1)

        x = self.activation_function(self.fc1(x))
        x = self.activation_function(self.fc2(x))
        x = self.activation_function(self.fc3(x))
        x = self.output_layer_activation_function(self.fc4(x))

        return x

    def get_norm(self) -> float:
        norm = 0.0

        for parameter in self.parameters():
            norm += torch.norm(parameter).item()

        return norm
