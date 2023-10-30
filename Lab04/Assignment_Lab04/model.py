import torch
import torch.nn as nn
from torch import Tensor

class BuildingsModel(nn.Module):
    def __init__(self, shape_of_image):
        super(BuildingsModel, self).__init__()

        self.x_dim = shape_of_image[0]
        self.y_dim = shape_of_image[1]
        self.z_dim = shape_of_image[2]

        input_channels = 3
        noConvFilters = 3
        convFilterSize = 5

        poolSize = 4

        inputSize = self.x_dim * ((((self.y_dim - (convFilterSize - 1)) // poolSize) - (convFilterSize - 1)) //
                                  poolSize) * ((((self.y_dim - (convFilterSize - 1)) // poolSize) -
                                                (convFilterSize - 1)) // poolSize) + 1

        outputSize = self.x_dim * self.y_dim * self.z_dim

        self.convLayer1 = nn.Conv2d(input_channels, noConvFilters, kernel_size=convFilterSize)
        self.poolLayer1 = nn.MaxPool2d(kernel_size=poolSize)

        self.convLayer2 = nn.Conv2d(input_channels, noConvFilters, kernel_size=convFilterSize)
        self.poolLayer2 = nn.MaxPool2d(kernel_size=poolSize)

        self.cLayer = nn.Linear(inputSize, outputSize)

    def forward(self, input):
        start_image, time_skip = input

        output = torch.relu(self.convLayer1(start_image))
        output = self.poolLayer1(output)
        output = torch.relu(self.convLayer2(output))
        output = self.poolLayer2(output)
        output = torch.flatten(output)

        output = torch.concat((output, Tensor([time_skip])))

        output = torch.tanh(self.cLayer(output))
        output = output.view((self.x_dim, self.y_dim, self.z_dim))
        output = output + start_image

        return torch.sigmoid(output)
