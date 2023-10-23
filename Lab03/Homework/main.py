from benchmarkers import accuracy, cross_entropy_loss
from network import epoch_train
from layer import Layer, InputLayer
from torchvision.datasets import MNIST
from torch import Tensor
import torch


def flatten_img(img_data: Tensor) -> Tensor:
    dims = img_data.size()
    assert len(dims) == 3
    return img_data.view(dims[0], dims[1] * dims[2])


def one_hot_encode(targets: Tensor) -> Tensor:
    repeat_targets = targets.view(-1, 1).repeat(1, 10)
    repeat_range = torch.arange(0, 10, 1).view(1, -1).repeat(targets.size(dim=0), 1)

    return (repeat_targets == repeat_range) * 1.0


def std_dev_scale(data_set: Tensor) -> Tensor:
    data_set = data_set * 1.0
    data_mean = torch.mean(data_set, dim=0)
    data_std = torch.std(data_set, dim=0)
    return torch.nan_to_num((data_set - data_mean) / data_std, nan=0)


def get_minst_data() -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
    data_set = MNIST("./", download=True)

    #scaled_data = std_dev_scale(flatten_img(data_set.data))
    scaled_data = flatten_img(data_set.data) * 1.0

    train_end_idx = int(0.75 * len(scaled_data))
    val_end_idx = int(0.85 * len(scaled_data))

    train_data = scaled_data[0:train_end_idx, :], one_hot_encode(data_set.targets[0:train_end_idx])
    val_data = scaled_data[train_end_idx:val_end_idx, :], one_hot_encode(data_set.targets[train_end_idx:val_end_idx])
    test_data = scaled_data[val_end_idx:, :], one_hot_encode(data_set.targets[val_end_idx:])

    return train_data, val_data, test_data


if __name__ == "__main__":
    train_set, validation_set, test_set = get_minst_data()

    input_layer = InputLayer(784)
    hidden_layer = Layer(100, "sigmoid", input_layer)
    output_layer = Layer(10, "softmax", hidden_layer)
    nn_network = output_layer.compile()

    epoch_train(train_set, validation_set, 0.001, 4, 20, nn_network, accuracy, cross_entropy_loss)
