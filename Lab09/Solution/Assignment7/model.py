import torch.nn as nn


class ImageTransformCNN(nn.Module):
    def __init__(self):
        super(ImageTransformCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.fc11 = nn.Linear(3 * 32 * 32, 32 * 32)
        self.fc22 = nn.Linear(32 * 32, 28 * 28)

        # First model used for training
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 28 * 28)

        self.relu = nn.ReLU()

    def forward(self, x):
        self.features(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc11(x))
        x = self.fc22(x)

        # x = self.relu(self.conv1(x))
        # x = self.relu(self.conv2(x))
        #
        # x = x.view(x.size(0), -1)
        #
        # x = self.relu(self.fc1(x))
        # x = self.fc2(x)

        x = x.view(x.size(0), 1, 28, 28)

        return x
