import torch


class HandMadeConv2d:
    def __init__(self, input_channels: int, output_channels: int, kernel_size: tuple[int, int]):
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernelHeight, self.kernelWidth = kernel_size

        self.weights = torch.randn(output_channels, input_channels, self.kernelHeight, self.kernelWidth)

    def cross_correlation2d(self, image: torch.tensor, kernel: torch.Tensor) -> torch.Tensor:
        image_height, image_width = image.shape
        output = torch.randn(image_height - self.kernelHeight + 1, image_width - self.kernelWidth + 1)
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                output[i, j] = (image[i:i + self.kernelHeight, j:j + self.kernelWidth] * kernel).sum()
        return output

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        # Similar with Conv 2d, we assume input of (N,C_in,H,W) and compute output of (N,C_out,H,W)
        no_images = batch.shape[0]

        new_batch_height = batch.shape[2] - self.kernelHeight + 1
        new_batch_width = batch.shape[3] - self.kernelWidth + 1
        result = torch.zeros(no_images, self.output_channels, new_batch_height, new_batch_width)

        for image_index in range(no_images):
            for output_index in range(self.output_channels):
                for input_index in range(self.input_channels):
                    result[image_index, output_index] += self.cross_correlation2d(
                        batch[image_index, input_index],
                        self.weights[output_index, input_index]
                    )

        return result


def main():
    no_images = 2
    input_channels = 3
    output_channels = 2
    kernel_size = (4, 5)

    batch = torch.randn(no_images, input_channels, *kernel_size)  # Input images

    custom_conv2d_layer = HandMadeConv2d(input_channels=input_channels, output_channels=output_channels,
                                         kernel_size=kernel_size)
    out = custom_conv2d_layer(batch)
    print((torch.nn.functional.conv2d(batch, custom_conv2d_layer.weights) - out).abs().max())


if __name__ == '__main__':
    main()
