import torch
import torch.nn as nn

__all__ = ['MosaicMLP']


class MosaicMLP(nn.Module):
    def __init__(self, device: torch.device, no_units_per_layer: list[int], output_activation=None):
        super(MosaicMLP, self).__init__()
        self.device = device
        non_blocking = (self.device.type == 'cuda')

        self.output_activation = output_activation if output_activation is not None else nn.Identity()

        self.no_layers = len(no_units_per_layer)
        self.layers = nn.ModuleList()
        for index in range(self.no_layers - 1):
            no_units_layer1 = no_units_per_layer[index]
            no_units_layer2 = no_units_per_layer[index + 1]
            self.layers.append(nn.Linear(no_units_layer1, no_units_layer2).to(self.device, non_blocking=non_blocking))
        self.no_layers -= 1

    def forward(self, x):
        x.to(self.device)
        # Previous layers
        for index in range(self.no_layers - 1):
            x = torch.relu(self.layers[index](x)).to(self.device)

        # Last layer
        # print(self.layers[self.no_layers - 1])
        x = self.layers[self.no_layers - 1](x).to(self.device)

        return self.output_activation(x).to(self.device)
