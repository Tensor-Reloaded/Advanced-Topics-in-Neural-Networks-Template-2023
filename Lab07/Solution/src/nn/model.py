import typing as t
import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    device: torch.device

    def __init__(
        self,
        input_size: int,
        output_size: int,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super(NeuralNetwork, self).__init__()
        self.device = device

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(256 * input_size, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, output_size)

        torch.nn.init.kaiming_uniform_(self.fc1.weight)
        torch.nn.init.kaiming_uniform_(self.fc2.weight)
        torch.nn.init.kaiming_uniform_(self.fc3.weight)

        self.activation_function = nn.ReLU()
        self.output_layer_activation_function = nn.Identity()
        self.to(device=self.device, non_blocking=self.device == "cuda")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(device=self.device, non_blocking=self.device == "cuda")

        x = self.pool(self.activation_function(self.conv1(x)))
        x = self.pool(self.activation_function(self.conv2(x)))
        x = self.pool(self.activation_function(self.conv3(x)))

        x = x.view(x.shape[0], -1)

        x = self.activation_function(self.fc1(x))
        x = self.dropout1(x)
        x = self.activation_function(self.fc2(x))
        x = self.dropout2(x)
        x = self.output_layer_activation_function(self.fc3(x))

        return x
