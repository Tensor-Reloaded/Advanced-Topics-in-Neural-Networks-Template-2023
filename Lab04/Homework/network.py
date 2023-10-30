import torch
import torch.nn.functional as F
from torch.nn import Conv2d, Linear, MaxPool2d, Module

class PredictionModel(Module):
    def __init__(self):
        super(PredictionModel, self).__init__()
        self.conv1 = Conv2d(3, 8, 3)
        self.pool = MaxPool2d(2, 2)
        self.conv2 = Conv2d(8, 32, 3)
        self.fc1 = Linear(32 * 30 * 30 + 1, 240)
        self.fc2 = Linear(240, 168)
        self.fc3 = Linear(168, 128 * 128 * 3)

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
        