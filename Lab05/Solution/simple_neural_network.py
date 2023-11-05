import torch
import utils


class SimpleNeuralNetwork(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(SimpleNeuralNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)
        self.relu = torch.nn.ReLU(inplace=True)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x

    @staticmethod
    def for_device(input_size: int, hidden_size: int, output_size: int) -> tuple["SimpleNeuralNetwork", torch.device]:
        device = utils.get_default_device()
        model = SimpleNeuralNetwork(input_size, hidden_size, output_size)
        return model.to(device), device
