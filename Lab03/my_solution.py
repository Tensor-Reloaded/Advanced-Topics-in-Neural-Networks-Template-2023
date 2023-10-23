from typing import Tuple
import numpy as np
import torch
from torch import Tensor
from math import sqrt
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


def activate(x: Tensor, layer: int) -> Tensor:
    # print("Shape of x", x.shape)
    if layer == 0:
        return x.sigmoid()
    return x.softmax(dim=1)


def sigmoid(x: Tensor) -> Tensor:
    return x.sigmoid()


def backward(x: Tensor, y: Tensor, y_hidden: Tensor, y_output: Tensor, w_output: Tensor) -> Tuple[
    list[Tensor], list[Tensor]]:
    delta_b_output = y_output - y
    delta_w_output = y_hidden.T @ delta_b_output

    delta_b_hidden = (delta_b_output @ w_output.T) * y_hidden * (1 - y_hidden)
    delta_w_hidden = x.T @ delta_b_hidden
    delta_b_hidden = delta_b_hidden.mean(dim=0)  # On column
    delta_b_output = delta_b_output.mean(dim=0)  # On column

    return [delta_w_hidden, delta_w_output], [delta_b_hidden, delta_b_output]


def train_batch(x: Tensor, y: Tensor, w_list: list[Tensor], b_list: list[Tensor], lr: float) -> Tuple[list[Tensor], list[Tensor]]:
    y_hidden = activate(forward(x, w_list[0], b_list[0]), layer=0)
    y_output = activate(forward(y_hidden, w_list[1], b_list[1]), layer=1)

    delta_w_list, delta_b_list = backward(x, y, y_hidden, y_output, w_list[1])

    for layer_index in range(2):
        w_list[layer_index] -= lr * delta_w_list[layer_index]
        b_list[layer_index] -= lr * delta_b_list[layer_index]
    return w_list, b_list


def train_epoch(data: Tensor, labels: Tensor, w_list: list[Tensor], b_list: list[Tensor],
                lr: float, batch_size: int) -> Tuple[list[Tensor], list[Tensor]]:
    non_blocking = w_list[0].device.type == 'cuda'

    for i in range(0, data.shape[0], batch_size):
        x = data[i: i + batch_size].to(w_list[0].device, non_blocking=non_blocking)
        y = labels[i: i + batch_size].to(w_list[0].device, non_blocking=non_blocking)
        w_list, b_list = train_batch(x, y, w_list, b_list, lr)
    return w_list, b_list


# tried it for hidden layer but results were not effective
def softmax_derivative(x):
    batch_size, vector_dim = x.shape
    I = torch.eye(vector_dim).unsqueeze(0).expand(batch_size, -1, -1)

    derivative = (I - x.view(batch_size, vector_dim, 1) @ x.view(batch_size, 1, vector_dim)) * x.view(batch_size,
                                                                                                      vector_dim, 1)
    return derivative


def evaluate(data: Tensor, labels: Tensor, w_list: list[Tensor], b_list: list[Tensor], batch_size: int,
             get_y_max=False) -> Tuple[float, float]:
    # Labels are not one hot encoded, because we do not need them as one hot.
    total_correct_predictions = 0
    total_len = data.shape[0]
    non_blocking = w_list[0].device.type == 'cuda'

    total_loss = 0
    no_batches = total_len / batch_size
    for i in range(0, total_len, batch_size):
        x = data[i: i + batch_size].to(w_list[0].device, non_blocking=non_blocking)
        y_max_value_indices = labels[i: i + batch_size].to(w_list[0].device, non_blocking=non_blocking)
        predicted_distribution = activate(
            forward(activate(forward(x, w_list[0], b_list[0]), layer=0), w_list[1], b_list[1]), layer=1)

        total_loss += functional.cross_entropy(predicted_distribution, y_max_value_indices) / batch_size
        # check torch.max documentation
        predicted_max_value, predicted_max_value_indices = torch.max(predicted_distribution, dim=1)
        # we check if the indices of the max value per line correspond to the correct label. We get a boolean mask
        # with True where the indices are the same, false otherwise
        if get_y_max:
            y_max_value, y_max_value_indices = torch.max(y_max_value_indices, dim=1)
        equality_mask = predicted_max_value_indices == y_max_value_indices
        # We sum the boolean mask, and get the number of True values in the mask. We use .item() to get the value out of
        # the tensor
        correct_predictions = equality_mask.sum().item()
        # correct_predictions = (torch.max(predicted_distribution, dim=1)[1] == y).sum().item()
        total_correct_predictions += correct_predictions

    value = total_correct_predictions / total_len
    # used avg of all batch losses
    return value, total_loss / no_batches


def train(epochs: int = 1000, device: torch.device = get_default_device()):
    print(f"Using device {device}")
    lr = 0.005
    pin_memory = device.type == 'cuda'
    # w_list = [torch.rand((784, 100), device=device) * 0.01,
    #           torch.rand((100, 10), device=device) * 0.01]
    # w_list = [torch.rand((784, 100), device=device) * 0.01,
    #            torch.rand((100, 10), device=device) * 0.01]

    w_list = [torch.rand((784, 100), device=device) * 0.01,
              torch.rand((100, 10), device=device) * 0.003]

    b_list = [torch.zeros((1, 100), device=device),
              torch.zeros((1, 10), device=device)]

    batch_size = 100
    eval_batch_size = 500
    data, labels = load_mnist(train=True, pin_memory=pin_memory)
    data_test, labels_test = load_mnist(train=False, pin_memory=pin_memory)

    epochs = tqdm(range(epochs))
    for _ in epochs:
        w_list, b_list = train_epoch(data, labels, w_list, b_list, lr, batch_size)

        accuracy_train, loss_train = evaluate(data, labels, w_list, b_list, eval_batch_size, get_y_max=True)
        accuracy_test, loss_test = evaluate(data_test, labels_test, w_list, b_list, eval_batch_size)

        if accuracy_train > 0.9 and accuracy_test > 0.9:
            if lr > 0.001:
                lr -= 0.00005
            elif lr > 0.0001:
                lr -= 0.00001
        epochs.set_postfix_str(
            f"accuracy_test = {accuracy_test}, loss_test = {loss_test}, accuracy_train= {accuracy_train}, loss_train= {loss_train}")


if __name__ == '__main__':
    train(500, torch.device('cpu'))
    train(500)
