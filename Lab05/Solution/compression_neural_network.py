import torch
import utils
import math


def split_matrix_in_chunks(matrix: torch.Tensor, chunk_shape: tuple[int, int]) -> torch.Tensor:
    n_rows, n_cols = chunk_shape

    z, h, w = matrix.size()
    assert h % n_rows == 0
    assert w % n_cols == 0
    return (matrix.reshape(z, h//n_rows, n_rows, -1, n_cols)
            .swapaxes(2, 3)
            .reshape(z, -1, n_rows, n_cols))


def recombine_chunks_into_matrix(chunks: torch.Tensor) -> torch.Tensor:
    z, nr_of_slice, h, w = chunks.size()

    nr_of_slice_sqrt = math.isqrt(nr_of_slice)

    return (chunks.reshape(z, nr_of_slice_sqrt, nr_of_slice_sqrt, h, w)
            .swapaxes(3, 2)
            .reshape(z, nr_of_slice_sqrt * h, nr_of_slice_sqrt * w))


class CompressionLayer(torch.nn.Module):
    def __init__(self, input_size: tuple[int, int], kernel_size: tuple[int, int], output_size: tuple[int, int]):
        super(CompressionLayer, self).__init__()

        assert input_size[0] == input_size[1]
        assert kernel_size[0] == kernel_size[1]

        self.input_size = input_size
        self.kernel_size = kernel_size
        self.output_kernel_size = output_size

        nr_of_kernels = (input_size[0] * input_size[1]) // (kernel_size[0] * kernel_size[1])
        self.kernels = [
            torch.nn.Linear(
                kernel_size[0] * kernel_size[1],
                self.output_kernel_size[0] * self.output_kernel_size[1]
            ) for _ in range(nr_of_kernels)]

        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor):
        results = []

        x_chunks = split_matrix_in_chunks(x.reshape(-1, self.input_size[0], self.input_size[1]), self.kernel_size)
        for it in range(len(self.kernels)):
            chunk = x_chunks[:, it, :, :].view(-1, self.kernel_size[0] * self.kernel_size[1])
            chunk = self.relu(self.kernels[it](chunk))
            chunk = chunk.reshape(-1, 1, self.output_kernel_size[0], self.output_kernel_size[1])
            results.append(chunk)

        recombined_matrix = recombine_chunks_into_matrix(torch.cat(results, dim=1))
        batches, h, w = recombined_matrix.size()

        return recombined_matrix.reshape(batches, h * w)


class CompressionNeuralNetwork(torch.nn.Module):
    def __init__(self, hidden_size: int, output_size: int):
        super(CompressionNeuralNetwork, self).__init__()

        self.fc1 = CompressionLayer((28, 28), (4, 4), (2, 2))
        self.fc2 = torch.nn.Linear(196, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, output_size)

        self.relu = torch.nn.ReLU(inplace=True)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x

    @staticmethod
    def for_device(hidden_size: int, output_size: int) -> tuple["CompressionNeuralNetwork", torch.device]:
        device = utils.get_default_device()
        model = CompressionNeuralNetwork(hidden_size, output_size)
        return model.to(device), device
