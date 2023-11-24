import torch
from torch import nn


class HandmadeConv2d:
    def __init__(self, weights):
        self.in_channels = weights.size()[1]
        self.out_channels = weights.size()[0]
        # verify the case in witch the kernel shape is (n,n) and only one n is given as parameter
        if len(weights.size()) < 4:
            self.kernel_size = weights.size()[2]
        else:
            self.kernel_size = (weights.size()[2], weights.size()[3])
        self.weights = weights

    def __call__(self, input_tensor):
        batch_size, _, input_height, input_width = input_tensor.size()
        if type(self.kernel_size) is tuple:
            kernel_height, kernel_width = self.kernel_size
        else:
            kernel_height = kernel_width = self.kernel_size
        output_height = input_height - kernel_height + 1
        output_width = input_width - kernel_width + 1

        output_tensor = torch.zeros(batch_size, self.out_channels, output_height, output_width, device=input_tensor.device)
        for o in range(self.out_channels):
            for i in range(output_height):
                for j in range(output_width):
                    input_slice = input_tensor[:, :, i:i + kernel_height, j:j + kernel_width]

                    output_tensor[:, o, i, j] = torch.sum(input_slice * self.weights[o])

        return output_tensor

