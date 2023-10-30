import torch
import torch.nn as nn

_all__ = ['ImagePredictMLP']

class ImagePredictMLP(nn.Module):
    def __init__(self, input_dim, output_dim, output_activation=None):
        super(ImagePredictMLP, self).__init__()
        self.output_activation = output_activation if output_activation is not None else nn.Identity()

        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2= nn.Linear(128, 64)
        self.fc3 = nn.Linear(65,output_dim)

    def forward(self, x, diff):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        x=torch.cat((diff.unsqueeze(1), x), dim=1)

        x = self.fc3(x)
        
        return self.output_activation(x)
    