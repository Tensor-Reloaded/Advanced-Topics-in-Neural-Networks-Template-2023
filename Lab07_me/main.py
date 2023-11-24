import torch
import torch.nn as nn

from Handmade_conv2d import Handmade_conv2d


def print_hi(name):
    inp = torch.randn(1, 3, 10, 12)  # Input image

    # kernel of size 4x5 , with 3 input channels and 2 output channels
    w = torch.randn(2, 3, 4, 5)  # Conv weights
    # Your implementation . Can be made differently , like only passing the kernel size  for example
    custom_conv2d_layer = Handmade_conv2d(weights=w)

    conv_layer = nn.Conv2d(in_channels=3, out_channels=2, kernel_size=(4, 5))
    conv_layer.weight.data = w

    output = conv_layer(inp)
    our_output = custom_conv2d_layer(inp)

    print((torch.nn.functional.conv2d(inp, w) - our_output).abs().max())


    # Print the input and output shapes
    print("Input shape:", inp.shape)
    print("Output shape:", output.shape)

    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

    # Print the input matrix
    print("\nInput matrix:")
    print(inp[0, 0, :, :])

    print("\n\n")
    # Print the kernel matrices for each output channel
    print("\nKernel matrices:")
    for i in range(conv_layer.out_channels):
        print(f"Kernel for output channel {i + 1}:")
        print(conv_layer.weight.data[i, 0, :, :])
        print("\n")

    print("\n\n")

    # Print the output matrices for each output channel
    print("\nOutput matrices:")
    for i in range(conv_layer.out_channels):
        print(f"Output for channel {i + 1}:")
        print(output[0, i, :, :])
        print("\n")

        print(f"Output for channel with custom layer {i + 1}:")
        print(our_output[0, i, :, :])
        print("\n")



    print("\n\n")

if __name__ == '__main__':
    print_hi('PyCharm')

