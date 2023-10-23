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


def activate_output(x: Tensor) -> Tensor:
    return x.softmax(dim=1)


def activate_hidden(x: Tensor) -> Tensor:
    return 1 / (1 + torch.exp(-x))


def backward(x: Tensor, y: Tensor, a_L: Tensor, a_H: Tensor, w_L: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    error_L = a_L - y
    a_H = a_H[:, :, None]
    error_L = error_L[:, :, None]
    delta_w_L = (a_H @ error_L.transpose(1, 2)).sum(dim=0)
    delta_b_L = error_L.mean(dim=0).reshape(-1)  # On column
    error_H = a_H * (1 - a_H) * torch.matmul(w_L, error_L)
    delta_w_H = (x[:, :, None] @ error_H.transpose(1, 2)).sum(dim=0)
    delta_b_H = error_H.mean(dim=0).reshape(-1)
    return delta_w_H, delta_b_H, delta_w_L, delta_b_L


def collate(x) -> Tensor:
    if isinstance(x, (tuple, list)):
        if isinstance(x[0], Tensor):
            return torch.stack(x)
        return torch.tensor(x)
    raise "Not supported yet"
    # see torch\utils\data\_utils\collate.py


def to_one_hot(x: Tensor) -> Tensor:
    return torch.eye(x.max() + 1)[x]


def train_batch(x: Tensor, y: Tensor, w_H: Tensor, w_L: Tensor, b_H: Tensor, b_L: Tensor, lr: float) -> \
        Tuple[Tensor, Tensor, Tensor, Tensor]:
    a_H = activate_hidden(forward(x, w_H, b_H))
    a_L = activate_output(forward(a_H, w_L, b_L))

    delta_w_H, delta_b_H, delta_w_L, delta_b_L = backward(x, y, a_L, a_H, w_L)
    w_H -= lr * delta_w_H
    w_L -= lr * delta_w_L
    b_H -= lr * delta_b_H
    b_L -= lr * delta_b_L
    return w_H, w_L, b_H, b_L


def train_epoch(data: Tensor, labels: Tensor, w_H: Tensor, w_L: Tensor, b_H: Tensor, b_L: Tensor, lr: float,
                batch_size: int) \
        -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    non_blocking = w_H.device.type == 'cuda'
    for i in range(0, data.shape[0], batch_size):
        x = data[i: i + batch_size].to(w_H.device, non_blocking=non_blocking)
        y = labels[i: i + batch_size].to(w_H.device, non_blocking=non_blocking)
        w_H, w_L, b_H, b_L = train_batch(x, y, w_H, w_L, b_H, b_L, lr)
    return w_H, w_L, b_H, b_L,


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


def evaluate(data: Tensor, labels: Tensor, w_H: Tensor, w_L: Tensor, b_H: Tensor, b_L: Tensor,
             batch_size: int) -> Tuple[float, float]:
    # Labels are not one hot encoded, because we do not need them as one hot.
    total_correct_predictions = 0
    total_len = data.shape[0]
    non_blocking = w_H.device.type == 'cuda'
    cross_entropy_loss = 0
    n_batches = 0
    for i in range(0, total_len, batch_size):
        x = data[i: i + batch_size].to(w_H.device, non_blocking=non_blocking)
        y = labels[i: i + batch_size].to(w_H.device, non_blocking=non_blocking)
        activation_H = activate_hidden(forward(x, w_H, b_H))
        predicted_distribution = activate_output(forward(activation_H, w_L, b_L))
        cross_entropy_loss += torch.nn.functional.cross_entropy(predicted_distribution, y)
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
        n_batches += 1

    return total_correct_predictions / total_len, cross_entropy_loss / n_batches


def train(epochs: int = 1000, device: torch.device = get_default_device()):
    print(f"Using device {device}")
    pin_memory = device.type == 'cuda'  # Check the provided references.
    w_H = torch.rand((784, 100), device=device)
    w_L = torch.rand((100, 10), device=device)
    b_H = torch.zeros((1, 100), device=device)
    b_L = torch.zeros((1, 10), device=device)
    lr = 0.0005
    batch_size = 100
    eval_batch_size = 500
    data, labels = load_mnist(train=True, pin_memory=pin_memory)
    data_test, labels_test = load_mnist(train=False, pin_memory=pin_memory)
    epochs = tqdm(range(epochs))
    for _ in epochs:
        w_H, w_L, b_H, b_L = train_epoch(data, labels, w_H, w_L, b_H, b_L, lr, batch_size)
        accuracy, loss = evaluate(data_test, labels_test, w_H, w_L, b_H, b_L, eval_batch_size)
        epochs.set_postfix_str(f"accuracy = {accuracy}, loss = {loss}")


if __name__ == '__main__':
    train(500)
