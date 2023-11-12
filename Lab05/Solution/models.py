import torch
import torch.nn as nn

__all__ = ['MLP']


class MLP(nn.Module):
    def __init__(self, device: torch.device, no_units_per_layer: list[int], dropout_per_layer: list[float],
                 output_activation=None):
        super(MLP, self).__init__()

        self.device = device
        self.no_units_per_layer = no_units_per_layer
        self.output_activation = output_activation if output_activation is not None else nn.Identity()

        alpha = 1.

        self.no_layers = len(no_units_per_layer)
        self.layers = nn.ModuleList()
        for index in range(self.no_layers - 1):
            no_units_layer1 = no_units_per_layer[index]
            no_units_layer2 = no_units_per_layer[index + 1]

            # TODO:Add Instance Normalization

            # Create linear layer and initialize weights and bias
            layer = nn.Linear(no_units_layer1, no_units_layer2)
            nn.init.kaiming_normal_(layer.weight)
            layer.bias.data.fill_(0)

            if index != self.no_layers - 2:
                layer = nn.Sequential(
                    nn.Dropout(p=dropout_per_layer[index]),
                    layer,
                    # nn.InstanceNorm1d(num_features=256, affine=True),
                    # nn.LeakyReLU(negative_slope=slope, inplace=True),
                    nn.ELU(alpha=alpha, inplace=True),
                    nn.BatchNorm1d(no_units_layer2)
                )

            self.layers.append(layer)

        self.no_layers -= 1

        self.to(device)

    def forward(self, x):
        # Previous layers
        for index in range(self.no_layers - 1):
            x = self.layers[index](x)

        # Last layer
        x = self.layers[self.no_layers - 1](x)

        return self.output_activation(x)
