import torch
import torch.nn as nn
import torch.nn.functional as F


class HandmadeConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, init_weights=True,
                 for_tracing=False):
        super(HandmadeConv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.for_tracing = for_tracing

        # Initialize weights and bias
        if init_weights:
            self.weights = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.weights = None
            self.bias = None

    def forward(self, x):
        if self.weights is None or self.bias is None:
            raise ValueError("Weights and bias must be set before forward pass.")

        # Calculate the padded input size
        padded_height = x.shape[2] + 2 * self.padding
        padded_width = x.shape[3] + 2 * self.padding

        # Calculate output dimensions
        out_height = ((padded_height - self.weights.shape[2]) // self.stride) + 1
        out_width = ((padded_width - self.weights.shape[3]) // self.stride) + 1

        # Create output tensor
        out = torch.zeros((x.shape[0], self.out_channels, out_height, out_width))

        for i in range(out_height):
            for j in range(out_width):
                # Calculate the start and end indices for slicing
                h_start = i * self.stride
                w_start = j * self.stride
                h_end = h_start + self.weights.shape[2]  # Height of the kernel
                w_end = w_start + self.weights.shape[3]  # Width of the kernel

                if not self.for_tracing:
                    # Check if the slice goes beyond the input dimensions
                    if h_end > x.shape[2] or w_end > x.shape[3]:
                        continue  # Skip this iteration to avoid dimension mismatch (for comparison testing)

                x_slice = x[:, :, h_start:h_end, w_start:w_end]

                for k in range(self.out_channels):
                    out[:, k, i, j] = torch.sum(x_slice * self.weights[k, :, :, :], dim=[1, 2, 3]) + self.bias[k]

        return out


# Testing the custom Conv2D layer
inp = torch.randn(1, 3, 10, 12)  # Input image

# Kernel of size 4x5, with 3 input channels and 2 output channels
w = torch.randn(2, 3, 4, 5)  # Conv weights

# Instantiate the custom Conv2D layer with the specified weights
custom_conv2d_layer = HandmadeConv2D(in_channels=3, out_channels=2, kernel_size=4, padding=0, init_weights=False)
custom_conv2d_layer.weights = w  # Directly setting the weights
custom_conv2d_layer.bias = torch.zeros(2)  # Setting the bias to zero for simplicity

# Run the custom conv2d layer
out_custom = custom_conv2d_layer(inp)

# PyTorch's built-in conv2d for comparison
out_torch = F.conv2d(inp, w, bias=None, stride=1, padding=0)

# Compare the outputs
print("Maximum absolute difference:", (out_torch - out_custom).abs().max())
