from typing import Tuple
import numpy as np
import torch
from torch import Tensor
from torchvision.datasets import MNIST
from tqdm import tqdm


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def forward(x: Tensor, w: Tensor, b: Tensor) -> Tensor:
    return x @ w + b


def activate(x: Tensor) -> Tensor:
    return x.softmax(dim=1)


def backward(x: Tensor, y: Tensor, y_hat: Tensor, hidden: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    error = y_hat - y
    # First, compute the gradients for the hidden layer.
    # dE/dwh = dE/dy_hat * dy_hat/dz * dz/dwh
    # dE/dwh = (y_hat - y) * x_hidden
    delta_whidden = hidden @ error
    delta_bhidden = error.mean(dim=0)  # On column
    # Second, compute the gradients for the input layer.
    # dE/dwi = dE/dy_hat * dy_hat/dz * dz/dwh * dwh/dwi
    # dE/dwi = delta_whidden * (y_hat - y) * x
    # de/dwi = delta_whidden * error * x
    delta_winput = (delta_whidden.T @ x).T @ error.T
    delta_binput = error.mean(dim=1)
    return delta_whidden, delta_bhidden, delta_winput, delta_binput


def collate(x) -> Tensor:
    if isinstance(x, (tuple, list)):
        if isinstance(x[0], Tensor):
            return torch.stack(x)
        return torch.tensor(x)
    raise "Not supported yet"


def to_one_hot(x: Tensor) -> Tensor:
    return torch.eye(x.max() + 1)[x]


def train_batch(x: Tensor, y: Tensor, w1: Tensor, b1: Tensor, w2: Tensor, b2:Tensor, lr: float) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    hidden = activate(forward(x, w1, b1))
    y_hat = activate(forward(hidden, w2, b2))
    delta_whidden, delta_bhidden, delta_winput, delta_binput = backward(x, y, y_hat, hidden)
    w1 -= lr * delta_winput
    b1 -= lr * delta_binput
    w2 -= lr * delta_whidden
    b2 -= lr * delta_bhidden
    return w1, b1, w2, b2


def train_epoch(data: Tensor, labels: Tensor, w1: Tensor, b1: Tensor, w2: Tensor, b2: Tensor, lr: float, batch_size: int) \
        -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    non_blocking = w1.device.type == 'cuda'
    for i in range(0, data.shape[0], batch_size):
        x = data[i: i + batch_size].to(w1.device, non_blocking=non_blocking)
        y = labels[i: i + batch_size].to(w1.device, non_blocking=non_blocking)
        w1, b1, w2, b2 = train_batch(x, y, w1, b1, w2, b2, lr)
    return w1, b1, w2, b2


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


def evaluate(data: Tensor, labels: Tensor, w1: Tensor, b1: Tensor, w2: Tensor, b2:Tensor, batch_size: int) -> float:
    total_correct_predictions = 0
    total_len = data.shape[0]
    non_blocking = w1.device.type == 'cuda'
    for i in range(0, total_len, batch_size):
        x = data[i: i + batch_size].to(w1.device, non_blocking=non_blocking)
        y = labels[i: i + batch_size].to(w1.device, non_blocking=non_blocking)
        hidden = activate(forward(x, w1, b1)) 
        predicted_distribution = activate(forward(hidden, w2, b2))
        predicted_max_value, predicted_max_value_indices = torch.max(predicted_distribution, dim=1)
        equality_mask = predicted_max_value_indices == y
        correct_predictions = equality_mask.sum().item()
        total_correct_predictions += correct_predictions

    return total_correct_predictions / total_len


def train(epochs: int = 1000, device: torch.device = get_default_device()):
    print(f"Using device {device}")
    pin_memory = device.type == 'cuda'  
    w1 = torch.rand((784, 100), device=device) # weights input layer
    w2 = torch.rand((100, 10), device=device) # weights hidden layer
    b1 = torch.zeros((1, 100), device=device) # biases input layer
    b2 = torch.zeros((1, 10), device=device) # biases hidden layer
    lr = 0.0005
    batch_size = 100
    eval_batch_size = 500
    data, labels = load_mnist(train=True, pin_memory=pin_memory)
    data_test, labels_test = load_mnist(train=False, pin_memory=pin_memory)
    epochs = tqdm(range(epochs))
    for _ in epochs:
        w1, b1, w2, b2 = train_epoch(data, labels, w1, b1, w2, b2, lr, batch_size)
        accuracy = evaluate(data_test, labels_test, w1, b1, w2, b2, eval_batch_size)
        epochs.set_postfix_str(f"accuracy = {accuracy}")


if __name__ == '__main__':
    train(1000, torch.device('mps'))
    train(1000)
