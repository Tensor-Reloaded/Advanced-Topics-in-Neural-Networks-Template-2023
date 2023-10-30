import torch
import torch.nn as nn

__all__ = ['GlobalImageMLP']


class GlobalImageMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GlobalImageMLP, self).__init__()

        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x
