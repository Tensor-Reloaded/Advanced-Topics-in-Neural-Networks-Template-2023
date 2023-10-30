import torch
import torch.nn as nn


class ImagePredictionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, output_activation=None, device="cpu"):
        super(ImagePredictionModel, self).__init__()
        # Define the layers
        self.non_blocking = device == "cuda"
        self.fc1 = nn.Linear(input_size - 1, hidden_size).to(device=device, non_blocking=self.non_blocking)
        self.relu = nn.ReLU().to(device=device, non_blocking=self.non_blocking)
        self.fc2 = nn.Linear(hidden_size + 1, output_size).to(device=device, non_blocking=self.non_blocking)
        self.output_activation = output_activation if output_activation else nn.Sigmoid()
        self.device = device

    def forward(self, x_image, x_days):
        # Combine the image data and days data
        # Forward pass
        x = self.fc1(x_image)
        x = self.relu(x)
        x = torch.cat((x, x_days.unsqueeze(1)), dim=1)
        x = self.fc2(x)
        output = self.output_activation(x)

        return output

# Example usage
# input_size = 10  # Example input size (adjust as needed)
# hidden_size = 128
# output_size = 10  # Example output size (same as input size for this illustration)
#
# model = ImagePredictionModel(input_size, hidden_size, output_size)
