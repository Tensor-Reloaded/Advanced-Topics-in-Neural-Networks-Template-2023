import typing as t
import torch
import numpy as np
import mnist_loader
import nn


def exercise_1b():
    mnist_loader_instance = mnist_loader.MnistDataloader(
        "../datasets/mnist/train-images.idx3-ubyte",
        "../datasets/mnist/train-labels.idx1-ubyte",
        "../datasets/mnist/t10k-images.idx3-ubyte",
        "../datasets/mnist/t10k-labels.idx1-ubyte",
    )
    (train_images, train_labels), (
        test_images,
        test_labels,
    ) = mnist_loader_instance.load_data()

    train_images = images_to_tensors(train_images)
    train_labels = labels_to_tensors(train_labels)
    test_images = images_to_tensors(test_images)
    test_labels = labels_to_tensors(test_labels)

    W, b, mu = init_neural_network()

    accuracy_before_training = benchmark(test_images, W, b, test_labels)
    print(f"Accuracy before training:\t{accuracy_before_training * 100:.5f}%")

    trained_W, trained_b = train_n_times(10, train_images, W, b, train_labels, mu)

    accuracy_after_training = benchmark(test_images, trained_W, trained_b, test_labels)
    print(f"Accuracy after training:\t{accuracy_after_training * 100:.5f}%")


def images_to_tensors(images: torch.Tensor) -> torch.Tensor:
    tensors = torch.from_numpy(np.array(images, dtype=np.float32))
    return tensors.view(tensors.shape[0], -1)


def labels_to_tensors(labels: list) -> torch.Tensor:
    labels_size = len(set(labels))
    data = [[1 if i == label else 0 for i in range(0, labels_size)] for label in labels]
    return torch.Tensor(data)

def init_neural_network() -> t.Tuple[torch.Tensor, torch.Tensor, float]:
    W = torch.rand(784, 10)
    b = torch.rand(10, 1)
    mu = 0.5

    return W, b, mu

def benchmark(
    X: torch.Tensor, W: torch.Tensor, b: torch.Tensor, y_true: torch.Tensor
) -> torch.float32:
    instances = y_true.shape[0]
    accurate_guesses = 0

    for entry, label in zip(X, y_true):
        result = nn.forward_propagation(entry.view(-1, 1), W, b)
        accurate_guesses += torch.argmax(result) == torch.argmax(label)

    return accurate_guesses / instances


def train_n_times(
    n: int,
    X: torch.Tensor,
    W: torch.Tensor,
    b: torch.Tensor,
    y_true: torch.Tensor,
    mu: torch.float32,
) -> t.Tuple[torch.Tensor, torch.Tensor]:
    current_W = W
    current_b = b

    for i in range(0, n):
        print(f"Training: \t{i + 1} / {n}", end="\r")

        for entry, label in zip(X, y_true):
            current_W, current_b = nn.backward_propagation(
                entry.view(-1, 1), current_W, current_b, label.view(-1, 1), mu
            )

    return current_W, current_b
