from torch import nn


class ConvNet(nn.Module):
    def __init__(self, input_channels, input_size, num_classes):
        super(ConvNet, self).__init__()
        self.input_size = input_size
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * (input_size // 2) * (input_size // 2), 512)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool(x)
        x = x.view(-1, 64 * (self.input_size // 2) * (self.input_size // 2))
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x