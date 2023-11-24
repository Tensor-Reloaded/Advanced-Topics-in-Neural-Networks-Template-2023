import torch
import torch.nn as nn
import torch.nn.functional as F
class Handmade_conv2d(nn.Module):

    def __init__(self, weights):
        super(Handmade_conv2d, self).__init__()

        self.kernel_size = (weights.shape[2], weights.shape[3])
        self.in_channels = weights.shape[1]
        self.out_channels = weights.shape[0]

        self.weights = weights


    def forward(self, input_tensor):
        assert input_tensor.shape[1] == self.in_channels, "Input channels do not match."

        # Get input dimensions
        batch_size, _, input_height, input_width = input_tensor.shape

        kernel_height, kernel_width = self.kernel_size

        # Calculate output dimensions
        output_height = input_height - kernel_height + 1
        output_width = input_width - kernel_width + 1

        output_tensor = torch.zeros(batch_size, self.out_channels, output_height, output_width)

        #padded_input = F.pad(input_tensor, (kernel_width // 2, kernel_width // 2, kernel_height // 2, kernel_height // 2))
        padded_input = input_tensor

        for batch in range(batch_size):

            for out_channel in range(self.out_channels):

                for i in range(output_height):
                    for j in range(output_width):

                        for in_channel in range(self.in_channels):
                            relevant_slice = padded_input[batch, :, i:i + kernel_height, j:j + kernel_width]
                            output_tensor[batch, out_channel, i, j] = torch.sum( relevant_slice * self.weights[out_channel, :, :, :])

        return output_tensor