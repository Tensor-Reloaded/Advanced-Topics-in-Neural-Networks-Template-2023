import torch


def sigmoid(x):
    return 1.0 / (1.0 + torch.exp(-x))

def softmax(z):
    aux = torch.exp(z)
    return aux / torch.sum(aux)

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))