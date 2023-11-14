from typing import List, Tuple

import torch
from torch import nn, Tensor

from Assignment5 import transforms
from Assignment5.utils import turn_to_zero


class Model(nn.Module):  # only builds linear layers since this is all we learned up to now;
    # otherwise an additional parameter specifying layer types would have been added (similar to the activations one)
    # (will probably be built for next homeworks)
    def __init__(self, input_dim: int, output_dim: int, hidden_layers: List[int] = None,
                 device: torch.device = torch.device('cpu'),
                 activations: List = None, loss=nn.BCELoss(), dropouts: List[Tuple[str, float]] = None,
                 activations_test: List = None,
                 weight_initialization=None, momentum: float = 0.001,
                 batch_normalization: bool = False, regularization=None,
                 gradient_clipping: bool = False, clip_value: float = 1.0, lr: float = 0.005,
                 optimizers: List = None, optimizer_args: List[dict] = None, default_optim_args: dict = None,
                 optim_layers: List = None, if_train=True,
                 lr_scheduler=None, closure: List[bool] = None,
                 batch_norms=None):
        super(Model, self).__init__()
        if not default_optim_args:
            default_optim_args = {'lr': 0.01}
        self.optimizers = optimizers
        self.optimizer_layers = optim_layers
        self.optimizer_args = optimizer_args
        self.default_optim_args = default_optim_args
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight_init = weight_initialization
        self.dropouts = dropouts
        self.batch_normalization = batch_normalization
        self.regularization = regularization
        self.gradient_clipping = gradient_clipping
        self.clip_value = clip_value
        self.activations = activations
        self.device = device
        self.lr = lr
        self.lr_scheduler = lr_scheduler
        self.closure = closure
        self.activations_test = activations_test
        if not self.activations_test:
            self.activations_test = list(self.activations)
        self.if_train = if_train
        self.batch_norms = nn.ModuleList(batch_norms)
        self.dropouts = nn.ModuleList(dropouts)

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
            if activations and len(activations) > index \
                    and activations[index] != -1:
                torch.nn.init.zeros_(self.layers[index].bias)

                # pay attention so that your weight_init function modifies its parameter

        self.loss = loss
        self.loss = (self.loss + momentum * regularization(self.layers)) if regularization else self.loss

    def forward(self, features):
        # ask for batch normalization if that's the case
        features_init = Tensor(features)
        # traverse the model and apply activations (and dropouts)
        if self.if_train:
            activations = list(self.activations)
        else:
            activations = list(self.activations_test)
        for index, layer in enumerate(self.layers[:-1]):
            if self.batch_normalization:
                features = self.batch_norms[index](features).to(self.device)
            features = layer(features)
            if activations and len(activations) > index \
                    and activations[index] != -1:
                features = activations[index](features)
            if features.shape == features_init.shape:
                features += features_init
                features_init = Tensor(features)
            if (self.dropouts and len(self.dropouts) > index
                    and self.dropouts[index] != -1):
                features = self.dropouts[index](features).to(self.device)
        if self.batch_normalization:
            features = self.batch_norms[-1](features).to(self.device)
        features = self.layers[-1](features)  # default identity activation if no other given
        if activations and len(activations) >= len(self.layers):
            features = activations[len(self.layers) - 1](features)
        if (self.dropouts and len(self.dropouts) >= len(self.layers)
                and self.dropouts[len(self.layers) - 1] != -1):
            features = self.dropouts[len(self.layers) - 1](features).to(self.device)
        return features

    def get_model_norm(self):
        norm = 0.0
        for param in self.parameters():
            norm += torch.norm(param)
        return norm
