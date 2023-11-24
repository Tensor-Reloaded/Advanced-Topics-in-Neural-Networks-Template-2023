import torch

class MyConv2d:
    def __init__(self, kernel_weights):
        self.kernel_weights = kernel_weights

    def __call__(self, input_tensor):
        batch_size, in_channels, in_height, in_width = input_tensor.shape
        out_channels, _, kernel_height, kernel_width = self.kernel_weights.shape

        out_height = in_height - kernel_height + 1
        out_width = in_width - kernel_width + 1

        output_tensor = torch.zeros((batch_size, out_channels, out_height, out_width))

        for batch_idx in range(batch_size):
            for row_idx in range(out_height):
                for col_idx in range(out_width):
                    input_region = input_tensor[batch_idx, :, row_idx:row_idx + kernel_height, col_idx:col_idx + kernel_width]
                    output_tensor[batch_idx, :, row_idx, col_idx] = torch.sum(input_region * self.kernel_weights, dim=(1, 2, 3))

        return output_tensor

input_image = torch.randn(1, 3, 10, 12)
conv_weights = torch.randn(2, 3, 4, 5)

custom_conv2d_layer = MyConv2d(kernel_weights=conv_weights)

output_handmade = custom_conv2d_layer(input_image)

output_pytorch = torch.nn.functional.conv2d(input_image, conv_weights)

max_absolute_difference = (output_pytorch - output_handmade).abs().max()
print(max_absolute_difference)