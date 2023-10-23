import torch
import torch.nn.functional as F


class Activations:
    @staticmethod
    def sigmoid(tensor):
        return 1.0 / (1.0 + torch.exp(-tensor))

    @staticmethod
    def sigmoid_derivative(tensor):
        return Activations.sigmoid(tensor) * (1 - Activations.sigmoid(tensor))

    @staticmethod
    def softmax(tensor):
        aux = torch.exp(tensor)
        return aux / aux.sum()
