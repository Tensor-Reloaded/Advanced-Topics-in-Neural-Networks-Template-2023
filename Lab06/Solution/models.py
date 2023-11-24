import torch
import torch.nn as nn

__all__ = ['MLP', 'CNN']


class MLP(nn.Module):
    def __init__(self, device: torch.device,
                 no_units_per_layer: list[int], dropout_per_layer: list[float],
                 skip_connections: list[tuple[int, int]],
                 output_activation=None):
        super(MLP, self).__init__()

        self.device = device
        self.no_units_per_layer = no_units_per_layer
        self.output_activation = output_activation if output_activation is not None else nn.Identity()

        alpha = 1.

        self.no_layers = len(no_units_per_layer)

        # Info about the skip connections given
        skip_connections = [(connection[0] - 1, connection[1] - 1) for connection in skip_connections]
        # Establish where to save the output of the linear layers
        input_vertexes = set([connection[0] for connection in skip_connections])
        self.save_layer_output = [(i in input_vertexes) for i in range(self.no_layers - 2)]

        # Establish for each layer which previous layers are connected to it
        self.has_connections = [[] for _ in range(self.no_layers - 1)]
        for connection in skip_connections:
            index_layer1, index_layer2 = connection
            self.has_connections[index_layer2].append(index_layer1)

        self.layers = nn.ModuleList()
        for index in range(self.no_layers - 1):
            no_units_layer1 = no_units_per_layer[index]
            no_units_layer2 = no_units_per_layer[index + 1]

            # TODO:Add Instance Normalization

            # Create linear layer and initialize weights and bias
            layer = nn.Linear(no_units_layer1, no_units_layer2)
            nn.init.kaiming_uniform_(layer.weight)
            layer.bias.data.fill_(0)

            if index != self.no_layers - 2:
                layer = nn.Sequential(
                    nn.Dropout(p=dropout_per_layer[index], inplace=True),
                    layer)
                self.layers.append(layer)

                self.layers.append(nn.Sequential(
                    nn.ELU(alpha=alpha, inplace=True),
                    nn.BatchNorm1d(no_units_layer2)
                ))
            else:
                self.layers.append(layer)

        self.no_layers -= 1

        self.to(device)

    def forward(self, x):
        saved_output = [0.0 for _ in range(self.no_layers - 1)]

        for index in range(self.no_layers - 1):
            x = self.layers[2 * index](x)

            for previous_layer_index in self.has_connections[index]:
                x += saved_output[previous_layer_index]

            x = self.layers[2 * index + 1](x)

            if self.save_layer_output[index]:
                saved_output[index] = x.detach().clone()

        # Last layer
        x = self.layers[-1](x)

        return self.output_activation(x)


class CNN(nn.Module):
    def __init__(self, device: torch.device, no_classes: int, output_activation=None):
        super(CNN, self).__init__()

        self.device = device
        self.output_activation = output_activation if output_activation is not None else nn.Identity()

        self.activation_function = nn.ReLU()
        self.layers = nn.Sequential(
            self.get_block(3, 6, kernel_size=7, paddings=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self.get_block(6, 12, 5, 2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self.get_block(12, 24, 3, 1),
            nn.Conv2d(24, 6, kernel_size=1),
            nn.Flatten(),
            nn.Linear(6 * 8 * 8, 1024), nn.BatchNorm1d(1024), nn.ELU(), nn.Dropout(0.5),
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ELU(), nn.Dropout(0.5),
            nn.Linear(512, no_classes)
        )
        # The pipeline applies softmax

        self.to(device)

    def get_block(self, in_channels, out_channels, kernel_size, paddings):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=paddings),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ELU()
        )

    def forward(self, x):
        x = self.layers(x)
        return self.output_activation(x)
