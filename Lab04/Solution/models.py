import torch
import torch.nn as nn

__all__ = ['ImageMLP']


class ImageMLP(nn.Module):
    def __init__(self, input_dim, output_dim, output_activation=None):
        super(ImageMLP, self).__init__()
        self.output_activation = output_activation if output_activation is not None else nn.Identity()

        self.fc1 = nn.Linear(input_dim, input_dim * 5)
        self.fc3 = nn.Linear(input_dim * 5, output_dim)  # Assuming wine quality is rated from 0 to 10

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc3(x)

        return self.output_activation(x)