import torch
import torch.nn as nn

__all__ = ['CifraMLP']


class CifraMLP(nn.Module):
    def __init__(self, input_dim, hidden_size1, hidden_size2, output_dim):
        super(CifraMLP, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
