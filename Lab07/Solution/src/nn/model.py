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

        self.conv1 = nn.Conv2d(
            3, 27, kernel_size=3, stride=1, padding=1, device=self.device
        )
        self.conv2 = nn.Conv2d(
            27, 81, kernel_size=3, stride=1, padding=1, device=self.device
        )
        self.conv3 = nn.Conv2d(
            81, 243, kernel_size=3, stride=1, padding=1, device=self.device
        )

        self.fc1 = nn.Linear(243 * input_size, 256).to(
            device=self.device, non_blocking=self.device == "cuda"
        )
        self.fc2 = nn.Linear(256, 128).to(
            device=self.device, non_blocking=self.device == "cuda"
        )
        self.fc3 = nn.Linear(128, 64).to(
            device=self.device, non_blocking=self.device == "cuda"
        )
        self.fc4 = nn.Linear(64, output_size)

        torch.nn.init.kaiming_uniform_(self.fc1.weight)
        torch.nn.init.kaiming_uniform_(self.fc2.weight)
        torch.nn.init.kaiming_uniform_(self.fc3.weight)
        torch.nn.init.kaiming_uniform_(self.fc4.weight)

        self.activation_function = nn.ReLU()
        self.output_layer_activation_function = (
            output_layer_activation_function
            if output_layer_activation_function is not None
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(device=self.device, non_blocking=self.device == "cuda")

        x = self.activation_function(self.conv1(x))
        x = self.activation_function(self.conv2(x))
        x = self.activation_function(self.conv3(x))

        x = x.view(x.shape[0], -1)

        x = self.activation_function(self.fc1(x))
        x = self.activation_function(self.fc2(x))
        x = self.activation_function(self.fc3(x))
        x = self.output_layer_activation_function(self.fc4(x))

        return x
