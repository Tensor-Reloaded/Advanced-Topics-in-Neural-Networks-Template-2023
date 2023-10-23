import torch
from torch import Tensor
from layer import Layer
from network import feed_forward


def accuracy(input_data: Tensor, target: Tensor, network: [Layer]) -> float:
    predicted = feed_forward(input_data, network)
    return torch.sum(torch.argmax(predicted, dim=1) == torch.argmax(target, dim=1)) / len(input_data)


def cross_entropy_loss(input_data: Tensor, target: Tensor, network: [Layer]) -> float:
    predicted = feed_forward(input_data, network)
    per_element_result = target * torch.log(predicted) + (1 - target) * torch.log(1 - predicted)
    return torch.sum(torch.sum(per_element_result, dim=1), dim=0) / (-len(input_data))

