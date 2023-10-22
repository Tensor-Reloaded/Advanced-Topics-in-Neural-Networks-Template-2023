from typing import Tuple
import numpy as np
import torch
from torch import Tensor
from torchvision.datasets import MNIST
from tqdm import tqdm
import torch.nn.functional as functional


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
        # For multi-gpu workstations, PyTorch will use the first available GPU (cuda:0), unless specified otherwise
        # (cuda:1).
    if torch.backends.mps.is_available():
        return torch.device('mos')
    return torch.device('cpu')


def collate(x) -> Tensor:
    if isinstance(x, (tuple, list)):
        if isinstance(x[0], Tensor):
            return torch.stack(x)
        return torch.tensor(x)
    raise "Not supported yet"
    # see torch\utils\data\_utils\collate.py


def to_one_hot(x: Tensor) -> Tensor:
    return torch.eye(x.max() + 1)[x]


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


def forward(x: Tensor, w: Tensor, b) -> Tensor:
    # print("Shape of x", x.shape)
    # print("Shape of b", b.shape)
    return x @ w + b


def activate(x: Tensor) -> Tensor:
    return x.softmax(dim=1)


def sigmoid(x: Tensor) -> Tensor:
    return torch.sigmoid(x)


def backward(x: Tensor, y: Tensor, y_hat: Tensor) -> Tuple[Tensor, Tensor]:
    error = y_hat - y

    delta_w = x.T @ error
    delta_b = error.mean(dim=0)  # On column

    return delta_w, delta_b


def train_batch(x: Tensor, y: Tensor, w_input: Tensor, b_input: Tensor, w_output: Tensor, b_output: Tensor,
                lr: float) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    z_first = forward(x, w_input, b_input)
    y_first = activate(z_first)
    y_hat = activate(forward(y_first, w_output, b_output))

    sd = softmax_derivative(y_first)
    delta_b_first = torch.squeeze(torch.bmm(((y_hat - y) @ w_output.T).unsqueeze(1), sd), dim=1)
    delta_w_first = x.T @ delta_b_first
    delta_b_first = delta_b_first.mean(dim=0)

    w_input -= lr * delta_w_first
    b_input -= lr * delta_b_first

    delta_w, delta_b = backward(y_first, y, y_hat)
    w_output -= lr * delta_w
    b_output -= lr * delta_b

    # print(w_input.shape)
    # print(y_hat.shape)
    # print(delta_w.shape)
    # print(x.shape)
    value = (y_hat - y) @ w_output.T * (y_first * y_first - 1)

    return w_input, b_input, w_output, b_output


def train_epoch(data: Tensor, labels: Tensor, w_input: Tensor, b_input: Tensor, w_output: Tensor, b_output: Tensor,
                lr: float, batch_size: int) \
        -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    non_blocking = w_input.device.type == 'cuda'
    for i in range(0, data.shape[0], batch_size):
        x = data[i: i + batch_size].to(w_input.device, non_blocking=non_blocking)
        y = labels[i: i + batch_size].to(w_input.device, non_blocking=non_blocking)
        w_input, b_input, w_output, b_output = train_batch(x, y, w_input, b_input, w_output, b_output, lr)
    return w_input, b_input, w_output, b_output


def softmax_derivative(x):
    batch_size, vector_dim = x.shape
    I = torch.eye(vector_dim).unsqueeze(0).expand(batch_size, -1, -1)

    derivative = (I - x.view(batch_size, vector_dim, 1) @ x.view(batch_size, 1, vector_dim)) * x.view(batch_size,
                                                                                                      vector_dim, 1)
    return derivative


def evaluate(data: Tensor, labels: Tensor, w_input: Tensor, b_input: Tensor, w_output: Tensor, b_output: Tensor,
             batch_size: int) -> float:
    # Labels are not one hot encoded, because we do not need them as one hot.
    total_correct_predictions = 0
    total_len = data.shape[0]
    non_blocking = w_input.device.type == 'cuda'
    for i in range(0, total_len, batch_size):
        x = data[i: i + batch_size].to(w_input.device, non_blocking=non_blocking)
        y = labels[i: i + batch_size].to(w_input.device, non_blocking=non_blocking)
        predicted_distribution = activate(forward(activate(forward(x, w_input, b_input)), w_output, b_output))
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

    value = total_correct_predictions / total_len
    # if value > 0.9:
    #     global lr
    #     global not_done
    #     if not_done:
    #         lr = lr / 10
    return value


lr = 0.03
not_done = True

def train(epochs: int = 1000, device: torch.device = get_default_device()):
    print(f"Using device {device}")
    pin_memory = device.type == 'cuda'  # Check the provided references.
    w_input = torch.rand((784, 100), device=device)
    w_output = torch.rand((100, 10), device=device)
    b_input = torch.zeros((1, 100), device=device)
    b_output = torch.zeros((1, 10), device=device)

    batch_size = 64
    eval_batch_size = 500
    data, labels = load_mnist(train=True, pin_memory=pin_memory)
    data_test, labels_test = load_mnist(train=False, pin_memory=pin_memory)

    epochs_copy = epochs
    epochs = tqdm(range(epochs))
    for _ in epochs:
        # if _ > 10 and lr > 0.003:
        #     lr -= 0.001
        w_input, b_input, w_output, b_output = train_epoch(data, labels, w_input, b_input, w_output, b_output, lr,
                                                           batch_size)
        accuracy = evaluate(data_test, labels_test, w_input, b_input, w_output, b_output, eval_batch_size)
        epochs.set_postfix_str(f"accuracy = {accuracy}")


if __name__ == '__main__':
    train(500, torch.device('cpu'))
    train(500)
