import torch
import torch.nn as nn
from torch import Tensor


class SatelliteConv(nn.Module):
    def __init__(self, image_dims: tuple[int, int, int]):
        super(SatelliteConv, self).__init__()

        self.image_dims = image_dims

        self.conv1 = nn.Conv2d(3, 3, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=3)

        self.conv2 = nn.Conv2d(3, 3, kernel_size=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(3, 3, kernel_size=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.layer1 = nn.Linear(244, image_dims[0] * image_dims[1] * image_dims[2])

    def forward(self, data: tuple[Tensor, int]):
        input_data, nr_of_months = data

        feed_data = torch.relu(self.conv1(input_data))
        feed_data = self.pool1(feed_data)

        feed_data = torch.relu(self.conv2(feed_data))
        feed_data = self.pool2(feed_data)

        feed_data = torch.relu(self.conv3(feed_data))
        feed_data = self.pool3(feed_data)

        feed_data = torch.flatten(feed_data)

        feed_data = torch.concat((feed_data, Tensor([nr_of_months])))

        feed_data = torch.tanh(self.layer1(feed_data))
        feed_data = feed_data.view(self.image_dims)
        feed_data = feed_data + input_data

        return torch.sigmoid(feed_data)
