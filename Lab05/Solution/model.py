import torch
import torch.nn as nn


class ImageModel(nn.Module):
    def __init__(self, inputDimensions, hiddenDimensions, outputDimensions):
        super(ImageModel, self).__init__()
        self.fc1 = nn.Linear(inputDimensions, hiddenDimensions)
        self.fc2 = nn.Linear(hiddenDimensions, outputDimensions)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))