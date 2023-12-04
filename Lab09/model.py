import torch.nn as nn
import torch.nn.functional as nnf
from torchvision.transforms import functional as F


class TransformationModel(nn.Module):
    def __init__(self):
        super(TransformationModel, self).__init__()
        # A simple convolutional layer that maintains channel size but allows learning
        self.conv = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # Learned transformation
        x = nnf.relu(self.conv(x))
        # Apply deterministic transformations
        x = F.resize(x, size=(28, 28), antialias=True)
        x = F.rgb_to_grayscale(x, num_output_channels=1)
        x = F.hflip(x)
        x = F.vflip(x)
        return x
