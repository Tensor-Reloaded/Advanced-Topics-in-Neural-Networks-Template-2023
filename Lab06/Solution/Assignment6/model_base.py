from functools import partial
from typing import List

import torch
from torch import nn, Tensor
from torch.nn import Sequential


class ModelB(nn.Module):  # only builds linear layers since this is all we learned up to now;
    # otherwise an additional parameter specifying layer types would have been added (similar to the activations one)
    # (will probably be built for next homeworks)
    def __init__(self, device: torch.device = torch.device('cpu'),
                 layers: List = None,
                 loss=nn.CrossEntropyLoss(), weight_initialization=None, to_initialize=None,
                 gradient_clipping: bool = False, clip_value: float = 1.0,
                 optimizers: List = None, optimizer_args: List[dict] = None,
                 default_optim_args: dict = None, optim_layers: List = None,
                 lr_scheduler=None, closure: List[bool] = None,
                 lr_scheduler_args: dict = None, lr: float = 0.1):
        super(ModelB, self).__init__()
        if not default_optim_args:
            default_optim_args = {'lr': 0.01}
        self.optimizers = optimizers
        self.optimizer_layers = optim_layers
        self.optimizer_args = optimizer_args
        self.default_optim_args = default_optim_args
        self.weight_init = weight_initialization
        self.gradient_clipping = gradient_clipping
        self.clip_value = clip_value
        self.device = device
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_args = lr_scheduler_args
        self.closure = closure
        self.to_initialize = to_initialize
        self.lr = lr

        if layers:
            self.layers = Sequential(*layers).to(self.device)

        if self.to_initialize:
            for index, layer in enumerate(self.to_initialize):
                if self.weight_init and len(self.weight_init) > index \
                        and self.weight_init[index] != -1:
                    self.weight_init[index](self.layers[layer].weight)
                    # pay attention so that your weight_init function modifies its parameter

        self.loss = loss

    def forward(self, features: Tensor):
        features = self.layers(features)
        return features

    def get_model_norm(self):
        norm = 0.0
        for param in self.parameters():
            norm += torch.norm(param)
        return norm
