import torch
import torch.nn as nn
import torch.nn.functional as F


class ImagePredictionModel(nn.Module):
    def __init__(self):
        super(ImagePredictionModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 3)
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 32, 3)
        self.fc1 = nn.Linear(32 * 30 * 30 + 1, 240)
        self.fc2 = nn.Linear(240, 168)
        self.fc3 = nn.Linear(168, 128 * 128 * 3)

    def forward(self, x, time):
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.pool(F.leaky_relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        time = time.unsqueeze(1)
        x = torch.cat((x, time), dim=1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        x = x.view(x.size(0), 3, 128, 128)
        return x
