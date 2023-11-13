import torch
import torch.nn as nn

def get_model_norm(model):
    norm = 0.0
    for param in model.parameters():
        norm += torch.norm(param)
    return norm

class Cifar10MLP(nn.Module):
    def __init__(self, inputDimensions, hiddenDimensions, outputDimensions):
        super(Cifar10MLP, self).__init__()
        self.fc1 = nn.Linear(inputDimensions, hiddenDimensions)
        self.fc2 = nn.Linear(hiddenDimensions, outputDimensions)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

