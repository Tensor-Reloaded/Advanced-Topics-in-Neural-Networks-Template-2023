import torch


class Handmade_conv2d_implementation(torch.nn.Module):
    def __init__(self, weights):
        super(Handmade_conv2d_implementation, self).__init__()
        self.weights = weights

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        _, _, filter_lines_nr, filter_cols_nr = self.weights.shape

        # unfold lines
        data = data.unfold(2, filter_lines_nr, 1)
        # unfold cols
        data = data.unfold(3, filter_cols_nr, 1)

        # arrange axis for batched matrix multiplication
        data = torch.permute(data, (0, 2, 3, 1, 4, 5))

        # bring data and weights to same dimensions
        data = data.reshape(data.shape[0], data.shape[1], data.shape[2], data.shape[3] * data.shape[4] * data.shape[5])
        weights = self.weights.reshape(self.weights.shape[0],
                                       self.weights.shape[1] * self.weights.shape[2] * self.weights.shape[3])
        data = data.reshape(data.shape[0], data.shape[1], data.shape[2], data.shape[3], 1)

        # exploit the fact that unfolds, weights and channels are arranged exactly the same and to dot product
        result = torch.matmul(weights, data)

        # bring result back to expected shape
        result = result.reshape(result.shape[0], result.shape[1], result.shape[2], result.shape[3])
        result = torch.permute(result, (0, 3, 1, 2))

        return result


def main():
    inp = torch.randn(1, 3, 10, 12)

    w = torch.randn(2, 3, 4, 5)

    custom_conv2d_layer = Handmade_conv2d_implementation(weights=w)

    out = custom_conv2d_layer(inp)

    print((torch.nn.functional.conv2d(inp, w) - out).abs().max())


if __name__ == "__main__":
    main()
