import torch
from torch.optim.lr_scheduler import StepLR


class Model(torch.nn.Module):
    def __init__(self, input_size, output_size, config):
        super(Model, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.dropout1 = torch.nn.Dropout(0.5)
        self.fc3 = torch.nn.Linear(256, 128)
        self.fc4 = torch.nn.Linear(128, output_size)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.bn1 = torch.nn.BatchNorm1d(256)

        if config.base_optimizer:
            self.optimizer = config.optimizer(self.parameters(), config.base_optimizer, lr=config.learning_rate, **config.additional_params)
        else:
            self.optimizer = config.optimizer(self.parameters(), lr=config.learning_rate, **config.additional_params)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.bn1(self.fc2(x)))
        x = self.dropout1(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout2(x)
        x = self.fc4(x)
        return x

    def get_norm(self):
        norm = 0.0
        for param in self.parameters():
            norm += torch.norm(param)
        return norm
