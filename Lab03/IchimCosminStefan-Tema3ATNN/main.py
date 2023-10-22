import torch
import numpy as np
from torch import Tensor
from torchvision.datasets import MNIST
from tqdm import tqdm


def load_mnist(path: str = "./data", train: bool = True, pin_memory: bool = True):
    mnist_raw = MNIST(path, download=True, train=train)
    mnist_data = []
    mnist_labels = []
    for image, label in mnist_raw:
        tensor = torch.from_numpy(np.array(image))
        mnist_data.append(tensor)
        mnist_labels.append(label)
    mnist_data = collate(mnist_data).float()
    mnist_data = mnist_data.flatten(start_dim=1)
    mnist_data /= mnist_data.max()
    mnist_labels = collate(mnist_labels)
    if train:
        mnist_labels = to_one_hot(mnist_labels)
    if pin_memory:
        return mnist_data.pin_memory(), mnist_labels.pin_memory()
    return mnist_data, mnist_labels


def collate(x) -> Tensor:
    if isinstance(x, (tuple, list)):
        if isinstance(x[0], Tensor):
            return torch.stack(x)
        return torch.tensor(x)
    raise "Not supported yet"


def to_one_hot(x: Tensor) -> Tensor:
    return torch.eye(x.max() + 1)[x]


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mos')
    return torch.device('cpu')


def forward(x: Tensor, w: Tensor, b: Tensor) -> Tensor:
    return x @ w + b


def activate(x: Tensor) -> Tensor:
    return torch.special.expit(x)


def train(epochs: int = 1000, device: torch.device = get_default_device()):
    pin_memory = device.type == 'cuda'
    w1 = (1 + 1) * torch.rand((784, 100), device=device) - 1
    b1 = torch.zeros((1, 100), device=device)
    w2 = (1 + 1) * torch.rand((100, 10), device=device) - 1
    b2 = torch.zeros((1, 10), device=device)
    learning_rate = 0.001
    batch_size = 2000
    eval_batch_size = 500
    data, labels = load_mnist(train=True, pin_memory=pin_memory)
    data_test, labels_test = load_mnist(train=False, pin_memory=pin_memory)
    epochs = tqdm(range(epochs))
    total_loss = 0
    for i in epochs:
        w1, b1, w2, b2, total_loss, current_loss = train_epoch(data, labels, w1, b1, w2, b2, learning_rate, batch_size, total_loss)
        accuracy = evaluate(data_test, labels_test, w1, b1, w2, b2, eval_batch_size)
        epochs.set_postfix_str(f"accuracy = {accuracy}, current_loss = {current_loss}, total_loss = {total_loss}")
        if i % 50 == 0:
            learning_rate *= 0.9


def evaluate(data: Tensor, expected_labels: Tensor, weights_middle_layer: Tensor, biases_middle_layer: Tensor,
                weights_final_layer: Tensor, biases_final_layer: Tensor, batch_size: int) -> float:
    total_correct_predictions = 0
    total_len = data.shape[0]
    non_blocking = weights_middle_layer.device.type == 'cuda'
    for i in range(0, total_len, batch_size):
        current_data = data[i: i + batch_size].to(weights_middle_layer.device, non_blocking=non_blocking)
        current_labels = expected_labels[i: i + batch_size].to(weights_middle_layer.device, non_blocking=non_blocking)

        middle_layer_activations = activate(forward(current_data, weights_middle_layer, biases_middle_layer))
        final_layer_activations = activate(forward(middle_layer_activations, weights_final_layer, biases_final_layer))

        predicted_max_value, predicted_max_value_indices = torch.max(final_layer_activations, dim=1)
        equality_mask = predicted_max_value_indices == current_labels
        correct_predictions = equality_mask.sum().item()
        total_correct_predictions += correct_predictions
    return total_correct_predictions / total_len


def train_epoch(data: Tensor, expected_labels: Tensor, weights_middle_layer: Tensor, biases_middle_layer: Tensor,
                weights_final_layer: Tensor, biases_final_layer: Tensor,
                learning_rate: float, batch_size: int, total_loss) -> (Tensor, Tensor, Tensor, Tensor):
    non_blocking = weights_middle_layer.device.type == 'cuda'
    for i in range(0, data.shape[0], batch_size):
        current_data = data[i: i + batch_size].to(weights_middle_layer.device, non_blocking=non_blocking)
        current_expected_labels = expected_labels[i: i + batch_size].to(weights_middle_layer.device, non_blocking=non_blocking)
        middle_layer_activations = activate(forward(current_data, weights_middle_layer, biases_middle_layer))
        final_layer_activations = activate(forward(middle_layer_activations, weights_final_layer, biases_final_layer))

        loss_value = torch.nn.functional.cross_entropy(final_layer_activations, current_expected_labels)
        total_loss += loss_value.item()

        error_final_layer = final_layer_activations - current_expected_labels
        error_middle_layer = middle_layer_activations * (1 - middle_layer_activations) * (weights_final_layer @ error_final_layer.T).T

        weights_final_layer -= learning_rate * (middle_layer_activations.T @ error_final_layer)
        biases_final_layer -= learning_rate * error_final_layer.sum(dim=0, keepdim=True)

        weights_middle_layer -= learning_rate * (current_data.T @ error_middle_layer)
        biases_middle_layer -= learning_rate * error_middle_layer.sum(dim=0, keepdim=True)
    return weights_middle_layer, biases_middle_layer, weights_final_layer, biases_final_layer, total_loss, loss_value


if __name__ == '__main__':
    train(300, get_default_device())
    train(300, torch.device('cpu'))