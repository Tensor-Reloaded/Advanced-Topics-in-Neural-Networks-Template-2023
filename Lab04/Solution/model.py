import torch
import torch.nn as nn


class ImageModel(nn.Module):
    def __init__(self, input_dim, output_dim, output_activation=None):
        super(ImageModel, self).__init__()
        self.output_activation = output_activation if output_activation is not None else nn.Identity()

        self.fc1 = nn.Linear(input_dim, output_dim // 8)
        self.fc3 = nn.Linear(output_dim // 8, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc3(x)

        return self.output_activation(x)