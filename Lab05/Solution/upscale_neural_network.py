import torch
import utils


class UpscaleNeuronLayer(torch.nn.Module):
    def __init__(self):
        super(UpscaleNeuronLayer, self).__init__()

        self.downscale = torch.nn.Linear(784, 196)
        self.normalization = torch.nn.BatchNorm1d(196)
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor):
        downscaled_x = self.normalization(self.relu(self.downscale(x)))
        up_scaled_x = downscaled_x.reshape(-1, 14, 14).repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)
        x = (up_scaled_x + x.reshape(-1, 28, 28)).reshape(-1, 784)
        return x


class UpscaleNeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(UpscaleNeuralNetwork, self).__init__()
        self.fc1 = UpscaleNeuronLayer()
        self.fc2 = UpscaleNeuronLayer()

        self.fc3 = torch.nn.Linear(784, 512)
        self.fc4 = torch.nn.Linear(512, 384)
        self.fc5 = torch.nn.Linear(384, 10)

        self.relu = torch.nn.ReLU(inplace=True)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)

        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.softmax(self.fc5(x))

        return x

    @staticmethod
    def for_device(input_size: int, hidden_size: int, output_size: int) -> tuple["UpscaleNeuralNetwork", torch.device]:
        device = utils.get_default_device()
        model = UpscaleNeuralNetwork()
        return model.to(device), device
