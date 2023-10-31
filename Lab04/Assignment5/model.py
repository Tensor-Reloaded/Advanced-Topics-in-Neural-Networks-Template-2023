from typing import List, Tuple

import torch
from torch import nn

from Assignment5 import transforms


class Model(nn.Module):  # only builds linear layers since this is all we learned up to now;
    # otherwise an additional parameter specifying layer types would have been added (similar to the activations one)
    # (will probably be built for next homeworks)
    def __init__(self, input_dim: int, output_dim: int, hidden_layers: List[int] = None,
                 device: torch.device = torch.device('cpu'),
                 activations: List = None, loss=nn.BCELoss(), dropouts: List[Tuple[str, int]] = None,
                 weight_initialization=None, momentum: float = 0.001,
                 batch_normalization: bool = False, regularization=None, gradient_clipping: bool = False,
                 optimizers: List = None, optimizer_args: List[dict] = None, default_optim_args: dict = None):
        super(Model, self).__init__()
        if not default_optim_args:
            default_optim_args = {'lr': 0.01}
        self.optimizers = optimizers
        self.optimizer_args = optimizer_args
        self.default_optim_args = default_optim_args
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight_init = weight_initialization
        self.dropouts = dropouts
        self.batch_normalization = batch_normalization
        self.regularization = regularization
        self.gradient_clipping = gradient_clipping
        self.activations = activations
        self.loss = loss
        self.loss = (self.loss + momentum * regularization) if regularization else self.loss
        self.device = device

        self.layers = nn.ModuleList()
        # build network's layered structure
        if hidden_layers and len(hidden_layers) > 0:
            self.layers.append(nn.Linear(input_dim, hidden_layers[0]).to(device))
            for index, layer in enumerate(hidden_layers[:-1]):
                self.layers.append(nn.Linear(layer, hidden_layers[index + 1]).to(device))
            self.layers.append(nn.Linear(hidden_layers[-1], output_dim).to(device))
        else:
            self.layers.append(nn.Linear(input_dim, output_dim).to(device))
        # weight initialization if that's the case;
        # bias initialization would be similar and yet has not been exemplified
        for index, layer in enumerate(self.layers):
            if self.weight_init and len(self.weight_init) > index \
                    and self.weight_init[index] != -1:
                self.weight_init[index](self.layers[index].weight)
                # pay attention so that your weight_init function modifies its parameter

    def forward(self, features):
        # ask for batch normalization of that's the case
        if self.batch_normalization:
            features = transforms.StandardNormalization(
                mean=features.mean(), std=features.std())(features)

        # traverse the model and apply activations and dropouts
        for index, layer in enumerate(self.layers[:-1]):
            features = layer(features)
            if (self.dropouts and len(self.dropouts) > index
                    and self.dropouts[index][0] == 'b'  # from before activation
                    and 0 < self.dropouts[index][1] < 1):
                features = nn.Dropout(self.dropouts[index][1])(features).to(self.device)
            if self.activations and len(self.activations) > index \
                    and self.activations[index] != -1:
                features = self.activations[index](features)
            else:
                features = torch.relu(features)  # default reLu activation if no other given
            if (self.dropouts and len(self.dropouts) > index
                    and self.dropouts[index][0] == 'a'  # from after activation
                    and 0 < self.dropouts[index][1] < 1):
                features = nn.Dropout(self.dropouts[index][1])(features).to(self.device)
        features = self.layers[-1](features)  # default identity activation if no other given
        if self.activations and len(self.activations) >= len(self.layers):
            features = self.activations[len(self.layers) - 1](features)
        return features
