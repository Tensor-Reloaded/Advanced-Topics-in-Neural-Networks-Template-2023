import torch
import torch.nn as nn

class CustomMLP(nn.Module):
    def __init__(self, inputDimensions, outputDimensions):
        super(CustomMLP, self).__init__()
        self.output_activation = nn.Sigmoid()
        self.fc1 = nn.Linear(inputDimensions, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, outputDimensions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        
        return 255*self.output_activation(x)
