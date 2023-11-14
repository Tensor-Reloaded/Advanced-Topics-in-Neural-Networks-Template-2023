import torch

class MLP(torch.nn.Module):
    def __init__(self, input_size, output_size, config):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, output_size)

        if config.base_optimizer:
            self.optimizer = config.optimizer(self.parameters(), config.base_optimizer, lr=config.learning_rate)
        else:
            self.optimizer = config.optimizer(self.parameters(), lr=config.learning_rate)


    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def get_norm(self):
        norm = 0.0
        for param in self.parameters():
            norm += torch.norm(param)
        return norm