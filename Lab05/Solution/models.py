import torch
import torch.nn as nn

__all__ = ['MLP']


class MLP(nn.Module):
    def __init__(self, device: torch.device, no_units_per_layer: list[int], output_activation=None):
        super(MLP, self).__init__()

        self.device = device
        self.no_units_per_layer = no_units_per_layer
        self.output_activation = output_activation if output_activation is not None else nn.Identity()

        self.no_layers = len(no_units_per_layer)
        self.layers = nn.ModuleList()
        for index in range(self.no_layers - 1):
            no_units_layer1 = no_units_per_layer[index]
            no_units_layer2 = no_units_per_layer[index + 1]
            self.layers.append(nn.Linear(no_units_layer1, no_units_layer2))
        self.no_layers -= 1

        self.relu = torch.nn.ReLU(inplace=True)

        self.to(device)

    def forward(self, x):
        # Previous layers
        for index in range(self.no_layers - 1):
            x = self.relu(self.layers[index](x))

        # Last layer
        x = self.layers[self.no_layers - 1](x)

        return self.output_activation(x)
