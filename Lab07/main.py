import torch

def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mos')
    return torch.device('cpu')

class HandmadeConv2DImplementation(torch.nn.Module):
    def __init__(self, weights):
        super(HandmadeConv2DImplementation, self).__init__()
        self.weights = weights

    def forward(self, input_image):
        batch_size, _, img_height, img_width = input_image.shape
        output_channels, _, kernel_height, kernel_width = self.weights.shape
        stride_width, stride_height = 1, 1

        # Calculate output feature map dimensions
        feature_map_width = img_width - kernel_width + 1
        feature_map_height = img_height - kernel_height + 1

        # I. Unfold: Extract sliding local blocks
        patches = input_image.unfold(2, kernel_height, stride_height).unfold(3, kernel_width, stride_width)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous().view(batch_size, feature_map_height, feature_map_width, -1)

        # Reshape the weights matrix for matrix multiplication
        patches_w = self.weights.view(output_channels, -1).t()

        # II. Perform matrix multiplication
        result = torch.matmul(patches, patches_w)

        # III. View the output shape (fold)
        result = result.view(batch_size, feature_map_height, feature_map_width, output_channels)
        result = result.permute(0, 3, 1, 2).contiguous()

        return result

def main(device=get_default_device()):
    # Example: Input image with 3 channels and size 10x12
    input_image = torch.randn(1, 3, 10, 12).to(device)

    # Example: Convolutional kernel with 3 input channels and 2 output channels, size 4x5
    conv_weights = torch.randn(2, 3, 4, 5).to(device)

    # Create and apply handmade convolutional layer
    custom_conv2d_layer = HandmadeConv2DImplementation(weights=conv_weights)
    output = custom_conv2d_layer(input_image)

    # Compare with PyTorch's conv2d
    max_abs_diff = (torch.nn.functional.conv2d(input_image, conv_weights) - output).abs().max()
    print("Max absolute difference:", max_abs_diff)

if __name__ == '__main__':
    main()
