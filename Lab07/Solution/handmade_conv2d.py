import torch
import torch.nn as nn


class HandmadeConv2d(nn.Module):
    def __init__(self, weights):
        super(HandmadeConv2d, self).__init__()
        self.weights = nn.Parameter(weights)

    def forward(self, x):
        batch_size, input_channels_no, input_height, input_width = x.size()
        output_channels, _, kernel_height, kernel_width = self.weights.size()

        output_height, output_width = input_height - kernel_height + 1, input_width - kernel_width + 1
        output = torch.zeros(batch_size, output_channels, output_height, output_width)

        for i in range(output_height):
            for j in range(output_width):
                region = x[:, :, i:i + kernel_height, j:j + kernel_width]
                output[:, :, i, j] = torch.sum(region * self.weights, dim=(1, 2, 3))

        return output


if __name__ == '__main__':
    inp = torch.randn(1, 3, 10, 12)
    w = torch.randn(2, 3, 4, 5)
    custom_conv2d_layer = HandmadeConv2d(weights=w)
    out = custom_conv2d_layer(inp)
    output_builtin = nn.functional.conv2d(inp, w)
    delta_max = torch.max(nn.functional.conv2d(inp, w) - out).abs().max()
    print(f"Delta max is {delta_max}")
    threshold = 1e-5
    print("Pass") if delta_max < threshold else print("Fail")
