from typing import Tuple
import numpy as np
import torch
from torch import Tensor
from torchvision.datasets import MNIST
from tqdm import tqdm


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
        # For multi-gpu workstations, PyTorch will use the first available GPU (cuda:0), unless specified otherwise
        # (cuda:1).
    if torch.backends.mps.is_available():
        return torch.device('mos')
    return torch.device('cpu')


def forward(x: Tensor, w: Tensor, b: Tensor) -> Tensor:
    return x @ w + b


def activate(x: Tensor) -> Tensor:
    return x.softmax(dim=1)


def backward(x: Tensor, y: Tensor, y_hat: Tensor) -> Tuple[Tensor, Tensor]:
    error = y_hat - y
    delta_w = x.T @ error
    delta_b = error.mean(dim=0)  # On column
    return delta_w, delta_b


def collate(x) -> Tensor:
    if isinstance(x, (tuple, list)):
        if isinstance(x[0], Tensor):
            return torch.stack(x)
        return torch.tensor(x)
    raise "Not supported yet"
    # see torch\utils\data\_utils\collate.py


def to_one_hot(x: Tensor) -> Tensor:
    return torch.eye(x.max() + 1)[x]


def train_batch(x: Tensor, y: Tensor, w: Tensor, b: Tensor, lr: float) -> Tuple[Tensor, Tensor]:
    y_hat = activate(forward(x, w, b))
    delta_w_t, delta_b = backward(x, y, y_hat)
    w -= lr * delta_w_t
    b -= lr * delta_b
    return w, b


def train_epoch(data: Tensor, labels: Tensor, w: Tensor, b: Tensor, lr: float, batch_size: int) \
        -> Tuple[Tensor, Tensor]:
    non_blocking = w.device.type == 'cuda'
    for i in range(0, data.shape[0], batch_size):
        x = data[i: i + batch_size].to(w.device, non_blocking=non_blocking)
        y = labels[i: i + batch_size].to(w.device, non_blocking=non_blocking)
        w, b = train_batch(x, y, w, b, lr)
    return w, b


def load_mnist(path: str = "./data", train: bool = True, pin_memory: bool = True):
    mnist_raw = MNIST(path, download=True, train=train)
    mnist_data = []
    mnist_labels = []
    for image, label in mnist_raw:
        tensor = torch.from_numpy(np.array(image))
        mnist_data.append(tensor)
        mnist_labels.append(label)

    mnist_data = collate(mnist_data).float()  # shape 60000, 28, 28
    mnist_data = mnist_data.flatten(start_dim=1)  # shape 60000, 784
    mnist_data /= mnist_data.max()  # min max normalize
    mnist_labels = collate(mnist_labels)  # shape 60000
    if train:
        mnist_labels = to_one_hot(mnist_labels)  # shape 60000, 10
    if pin_memory:
        return mnist_data.pin_memory(), mnist_labels.pin_memory()
    return mnist_data, mnist_labels


def evaluate(data: Tensor, labels: Tensor, w: Tensor, b: Tensor, batch_size: int) -> float:
    # Labels are not one hot encoded, because we do not need them as one hot.
    total_correct_predictions = 0
    total_len = data.shape[0]
    non_blocking = w.device.type == 'cuda'
    for i in range(0, total_len, batch_size):
        x = data[i: i + batch_size].to(w.device, non_blocking=non_blocking)
        y = labels[i: i + batch_size].to(w.device, non_blocking=non_blocking)
        predicted_distribution = activate(forward(x, w, b))
        # check torch.max documentation
        predicted_max_value, predicted_max_value_indices = torch.max(predicted_distribution, dim=1)
        # we check if the indices of the max value per line correspond to the correct label. We get a boolean mask
        # with True where the indices are the same, false otherwise
        equality_mask = predicted_max_value_indices == y
        # We sum the boolean mask, and get the number of True values in the mask. We use .item() to get the value out of
        # the tensor
        correct_predictions = equality_mask.sum().item()
        # correct_predictions = (torch.max(predicted_distribution, dim=1)[1] == y).sum().item()
        total_correct_predictions += correct_predictions

    return total_correct_predictions / total_len


def train(epochs: int = 1000, device: torch.device = get_default_device()):
    print(f"Using device {device}")
    pin_memory = device.type == 'cuda'  # Check the provided references.
    w = torch.rand((784, 10), device=device)
    b = torch.zeros((1, 10), device=device)
    lr = 0.0005
    batch_size = 100
    eval_batch_size = 500
    data, labels = load_mnist(train=True, pin_memory=pin_memory)
    data_test, labels_test = load_mnist(train=False, pin_memory=pin_memory)
    epochs = tqdm(range(epochs))
    for _ in epochs:
        w, b = train_epoch(data, labels, w, b, lr, batch_size)
        accuracy = evaluate(data_test, labels_test, w, b, eval_batch_size)
        epochs.set_postfix_str(f"accuracy = {accuracy}")


if __name__ == '__main__':
    train(500, torch.device('cpu'))
    train(500)
