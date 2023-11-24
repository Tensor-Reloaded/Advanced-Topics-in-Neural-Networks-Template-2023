import torch


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mos')
    return torch.device('cpu')


class Handmade_conv2d_implementation(torch.nn.Module):
    def __init__(self, weights):
        super(Handmade_conv2d_implementation, self).__init__()
        self.weights = weights

    def forward(self, input_image):
        # The input in this particular case will consist of only one image (to be more specific - a tensor) and the
        # weights
        input_dimensions = input_image.shape
        batch_size = input_dimensions[0]  # The value is 1
        img_height = input_dimensions[2]  # The height of image (number of lines) is 10
        img_width = input_dimensions[3]  # The width of image (number of columns) is 12

        filter_dimensions = self.weights.shape
        output_channels = filter_dimensions[0]  # The value is 2
        kernel_height = filter_dimensions[2]  # The height of kernel (number of lines) is 4
        kernel_width = filter_dimensions[3]  # The width of kernel (number of columns) is 5
        stride_width = 1
        stride_height = 1

        # The convolutional layer generates 2 feature maps, applying 2 convolutional filters of size 4x5 on the input
        # image. After applying the filters on the input image of size 10x12 (the number of channels was not included),
        # 2 feature maps of size 7x8 will result. Pixels are lost in the extremities (3 pixels per height and 4 pixels
        # per width) because the convolution is performed with a 4x5 filter without padding or other additional
        # operations.
        feature_map_width = img_width - kernel_width + 1
        feature_map_height = img_height - kernel_height + 1

        # Convolution is equivalent with Unfold + Matrix Multiplication + Fold (or view to output shape)

        # I. Unfold: extract sliding local blocks (using the unfold function is like going through the blocks in the
        # initial image over which I want to superimpose the convolution filter)
        patches = input_image.unfold(2, kernel_height, stride_height).unfold(3, kernel_width, stride_width)
        # Create 60 blocks of dimension 7x8
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous().view(batch_size, feature_map_height, feature_map_width,
                                                                      -1)  # The last 3 dimensions are flattened

        # Reshape the matrix of weights accordingly in order to perform matrix multiplication
        patches_w = self.weights.view(output_channels, -1).t()

        # II. Perform matrix multiplication
        result = torch.matmul(patches, patches_w)

        # III. View the output shape (fold)
        result = result.view(batch_size, feature_map_height, feature_map_width, output_channels)
        result = result.permute(0, 3, 1, 2).contiguous()

        return result


def main(device=get_default_device()):
    # Initial image has 3 input channels and its size is 10x12
    inp = torch.randn(1, 3, 10, 12).to(device)  # Input image

    # Kernel of size 4x5, with 3 input channels and 2 output channels
    w = torch.randn(2, 3, 4, 5).to(device)  # Conv weights

    # The result will consist of 2 feature maps with dimension 7x8 corresponding to those 2 convolution filters
    custom_conv2d_layer = Handmade_conv2d_implementation(weights=w)
    out = custom_conv2d_layer(inp)
    print((torch.nn.functional.conv2d(inp, w) - out).abs().max())


if __name__ == '__main__':
    main()
