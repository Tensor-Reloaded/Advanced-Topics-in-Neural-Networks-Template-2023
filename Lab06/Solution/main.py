import torch


class HandmadeConv2dImplementation(torch.nn.Module):
    def __init__(self, weights):
        super(HandmadeConv2dImplementation, self).__init__()
        self.weights = weights

    def forward(self, x):
        batch_size, in_channels, in_height, in_width = x.size()
        out_channels, _, kernel_height, kernel_width = self.weights.size()

        # Initialize the output tensor
        out_height = in_height - kernel_height + 1
        out_width = in_width - kernel_width + 1
        output = torch.zeros(batch_size, out_channels, out_height, out_width)

        # Perform the convolution
        for b in range(batch_size):
            for c_out in range(out_channels):
                for h_out in range(out_height):
                    for w_out in range(out_width):
                        # Extract the current patch from the input
                        patch = x[
                            b,
                            :,
                            h_out : h_out + kernel_height,
                            w_out : w_out + kernel_width,
                        ]

                        # Perform element-wise multiplication with the weights and sum
                        output[b, c_out, h_out, w_out] = (
                            patch * self.weights[c_out]
                        ).sum()

        return output


# Testing the custom Conv2D layer
inp = torch.randn(1, 3, 10, 12)  # Input image
w = torch.randn(2, 3, 4, 5)  # Conv weights

# Instantiate the handmade Conv2D layer
custom_conv2d_layer = HandmadeConv2dImplementation(weights=w)

# Forward pass through the handmade Conv2D layer
out_custom = custom_conv2d_layer(inp)

# Forward pass through the PyTorch built-in Conv2D layer
out_builtin = torch.nn.functional.conv2d(inp, w)

# Print the maximum absolute difference between the two outputs
print((out_builtin - out_custom).abs().max())
