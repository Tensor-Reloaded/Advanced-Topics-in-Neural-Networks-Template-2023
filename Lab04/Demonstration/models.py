import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, inputDimensions, outputDimensions):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(inputDimensions, outputDimensions // 5)
        self.fc2 = nn.Linear(outputDimensions // 5, outputDimensions // 10)
        self.fc3 = nn.Linear(outputDimensions // 10, outputDimensions)
        self.output_activation = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return 255 * self.output_activation(x)