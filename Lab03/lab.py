from typing import Tuple
import numpy as np
import torch
from torch import Tensor
from torchvision.datasets import MNIST
from tqdm import tqdm

# Helper function to select device base runner
def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

# Helper function to collate data into a tensor
def collate(x) -> Tensor:
    if isinstance(x, (tuple, list)):
        if isinstance(x[0], Tensor):
            return torch.stack(x)
        return torch.tensor(x)
    raise Exception("Not supported yet")


# Load MNIST dataset and convert it to tensors
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


# Convert labels to one-hot encoded tensors
def to_one_hot(x: Tensor) -> Tensor:
    return torch.eye(x.max() + 1)[x]


# Compute the forward pass in the neural network
def forward(x: Tensor, w: Tensor, b: Tensor) -> Tensor:
    return x @ w + b


# Activation function to convert raw scores into probabilities
def activate_softmax(x: Tensor) -> Tensor:
    return x.softmax(dim=1)


# Activation function used in the hidden layer
def activate_sigmoid(x: Tensor) -> Tensor:
    return torch.sigmoid(x)


# Train the model for one epoch and update the weights
def train_epoch(data: Tensor, labels: Tensor, w_hidden_layer: Tensor, b_hidden_layer: Tensor, w_last_layer: Tensor, b_last_layer: Tensor, lr: float, batch_size: int) \
        -> Tuple[Tensor, Tensor, Tensor, Tensor, float]:
    epoch_loss = 0
    non_blocking = w_hidden_layer.device.type == 'cuda'
    for i in range(0, data.shape[0], batch_size):
        x = data[i: i + batch_size].to(w_hidden_layer.device, non_blocking=non_blocking)
        y = labels[i: i + batch_size].to(w_hidden_layer.device, non_blocking=non_blocking)

        # Compute forward pass through hidden and output layers
        hidden_output = activate_sigmoid(forward(x, w_hidden_layer, b_hidden_layer))
        last_output = activate_softmax(forward(hidden_output, w_last_layer, b_last_layer))

        # Compute loss
        epoch_loss += torch.nn.functional.cross_entropy(last_output, y).item()

        # Backpropagation
        last_error = last_output - y
        delta_w_last = hidden_output.T @ last_error
        delta_b_last = last_error.mean(dim=0)
        hidden_error = (hidden_output * (1 - hidden_output)) * (w_last_layer @ last_error.T).T
        delta_w_hidden = x.T @ hidden_error
        delta_b_hidden = hidden_error.mean(dim=0)

        # Update weights and biases
        w_last_layer -= lr * delta_w_last
        b_last_layer -= lr * delta_b_last
        w_hidden_layer -= lr * delta_w_hidden
        b_hidden_layer -= lr * delta_b_hidden

    return w_hidden_layer, b_hidden_layer, w_last_layer, b_last_layer, epoch_loss / batch_size


# Evaluate the model on the test data and compute the accuracy
def evaluate(data: Tensor, labels: Tensor, w_hidden_layer: Tensor, b_hidden_layer: Tensor, w_last_layer: Tensor, b_last_layer: Tensor, batch_size: int) -> float:
    total_correct_predictions = 0
    total_len = data.shape[0]
    non_blocking = w_hidden_layer.device.type == 'cuda'
    for i in range(0, total_len, batch_size):
        x = data[i: i + batch_size].to(w_hidden_layer, non_blocking=non_blocking)
        y = labels[i: i + batch_size].to(w_hidden_layer.device, non_blocking=non_blocking)

        # Compute forward pass and predictions
        hidden_output = activate_sigmoid(forward(x, w_hidden_layer, b_hidden_layer))
        last_output = activate_softmax(forward(hidden_output, w_last_layer, b_last_layer))
        _, predicted_max_value_indices = torch.max(last_output, dim=1)

        # Count correct predictions
        equality_mask = predicted_max_value_indices == y
        correct_predictions = equality_mask.sum().item()
        total_correct_predictions += correct_predictions

    return total_correct_predictions / total_len


# The main function to train the model for several epochs
def train(epochs: int = 1000, device: torch.device = get_default_device()):
    print(f"Using device {device}")
    pin_memory = device.type == 'cuda'

    # Initialize weights and biases with normal distribution
    w_hidden_layer = torch.empty((784, 100), device=device).normal_(mean=0, std=np.power(np.sqrt(784), (-1)))
    b_hidden_layer = torch.empty((1, 100), device=device).normal_(mean=0, std=1)
    w_last_layer = torch.empty((100, 10), device=device).normal_(mean=0, std=np.power(np.sqrt(100), (-1)))
    b_last_layer = torch.empty((1, 10), device=device).normal_(mean=0, std=1)

    lr = 0.005  # Learning rate
    batch_size = 500  # Batch size for training
    eval_batch_size = 500  # Batch size for evaluation
    data, labels = load_mnist(train=True, pin_memory=pin_memory)
    data_test, labels_test = load_mnist(train=False, pin_memory=pin_memory)

    # Training loop
    epochs = tqdm(range(epochs))
    total_loss = 0
    for _ in epochs:
        # Train for one epoch
        w_hidden_layer, b_hidden_layer, w_last_layer, b_last_layer, epoch_loss = \
            train_epoch(data, labels, w_hidden_layer, b_hidden_layer, w_last_layer, b_last_layer, lr, batch_size)

        total_loss += epoch_loss

        # Evaluate the model
        accuracy = evaluate(data_test, labels_test, w_hidden_layer, b_hidden_layer, w_last_layer, b_last_layer, eval_batch_size)
        epochs.set_postfix_str(f"accuracy = {accuracy}, epoch loss = {epoch_loss}, total loss = {total_loss}")

        if accuracy > 0.9:
            if lr > 0.001:
                lr -= 0.00005
            elif lr > 0.0001:
                lr -= 0.00001

# Main script that calls the train function to start the training process
if __name__ == '__main__':
    train(500, torch.device('cpu'))  # Training on CPU
    train(500, get_default_device())  # Training on default

