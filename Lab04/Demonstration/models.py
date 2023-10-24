import torch
import torch.nn as nn


__all__ = ['WineQualityMLP']

class WineQualityMLP(nn.Module):
    def __init__(self, input_dim, output_dim, output_activation=None):
        super(WineQualityMLP, self).__init__()
        self.output_activation = output_activation if output_activation is not None else nn.Identity()
        
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)  # Assuming wine quality is rated from 0 to 10

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        
        return self.output_activation(x)