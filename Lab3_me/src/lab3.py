from typing import Tuple

import numpy as np

import torch
from torch import Tensor
from torchvision.datasets import MNIST
from tqdm import tqdm

print(torch.cuda.is_available())

##Dataset manipulation
def collate(x) -> Tensor:
    if isinstance(x, (tuple, list)):
        if isinstance(x[0], Tensor):
            return torch.stack(x)
        return torch.tensor(x)
    raise "Not supported yet"
    # see torch\utils\data\_utils\collate.py

def to_one_hot(x: Tensor) -> Tensor:
    num_classes = x.max() + 1
    return torch.eye(num_classes)[x]  #This essentially creates a matrix where each row corresponds to a class, and there is one row for each possible class.

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


##Machine learning
def activate(x: Tensor) -> Tensor:
    return x.softmax(dim=1)

def forward(x: Tensor, w: Tensor, b: Tensor) -> Tensor:
    #x - [batch_size, input_features]
    #w - [input_features, output_features]
    #b - [1, output_features[
    #x @ w + b - [ batch_size, output_features ]
    return x @ w + b

def backward(x: Tensor, y: Tensor, y_hat: Tensor) -> Tuple[Tensor, Tensor]:
    #x - [batch_size, input_features]
    #y - [batch_size, output_features]
    #y_hat - [batch_size, output_features]
    error = y_hat - y
    delta_w = x.T @ error  #- [input_features, output_features]
    delta_b = error.mean(dim=0)  # On column
    return delta_w, delta_b

def train_batch(x: Tensor, y: Tensor, w_hidden: Tensor, b_hidden: Tensor, w_output:Tensor, b_output: Tensor, lr: float) -> Tuple[Tensor, Tensor]:

    hidden_step = forward(x, w_hidden, b_hidden)
    hidden_activation = activate(hidden_step)

   # dropout_prob = 0.05
    #dropout_mask = (torch.rand_like(hidden_activation) > dropout_prob).float()
    #hidden_activation = hidden_activation * dropout_mask / (1 - dropout_prob)

    output_step = forward(hidden_activation, w_output, b_output)
    output_activation = activate(output_step)



    # Backpropagation starting from the back
    delta_w_output, delta_b_output = backward(hidden_activation, y, output_activation)

    #for hidden layer
    output_error = output_activation - y
    hidden_error = output_error @ w_output.T
    delta_w_hidden, delta_b_hidden = backward(x, hidden_error, hidden_activation)

    # Update the weights and biases using the calculated gradients
    w_output_t = w_output + lr * delta_w_output
    b_output_t = b_output + lr * delta_b_output

    w_hidden_t = w_hidden + lr * delta_w_hidden
    b_hidden_t = b_hidden + lr * delta_b_hidden

    return w_hidden_t,b_hidden_t, w_output_t, b_output_t

def evaluate(data: Tensor, labels: Tensor,
             w_hidden:Tensor, b_hidden: Tensor,
             w_output:Tensor, b_output: Tensor,
             batch_size: int) -> float:

    # Labels are not one hot encoded, because we do not need them as one hot.
    total_correct_predictions = 0
    total_len = data.shape[0]

    device = w_hidden.device
    non_blocking = device.type == 'cuda'

    for i in range(0, total_len, batch_size):
        x = data[i: i + batch_size].to(device, non_blocking=non_blocking)
        y = labels[i: i + batch_size].to(device, non_blocking=non_blocking)

        hidden_step = activate(forward(x, w_hidden, b_hidden))
        output_step = activate(forward(hidden_step, w_output, b_output))

        predicted_distribution = output_step

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

def train_epoch(data: Tensor, labels: Tensor,
                w_hidden: Tensor, b_hidden: Tensor,
                w_output: Tensor, b_output: Tensor,
                lr: float, batch_size: int
                ) -> Tuple[Tensor, Tensor]:

    device = w_hidden.device
    non_blocking = device.type == 'cuda'

    for i in range(0, data.shape[0], batch_size):
        #create one such batch
        x = data[i: i + batch_size].to(device, non_blocking=non_blocking)
        y = labels[i: i + batch_size].to(device, non_blocking=non_blocking)

        w_hidden, b_hidden, w_output, b_output = train_batch(x, y, w_hidden, b_hidden, w_output, b_output, lr)

    return w_hidden, b_hidden, w_output, b_output

def train():
    my_device = torch.device('cuda')

    input_size = 784
    hidden_size = 100
    output_size = 10

    w_hidden = torch.rand((input_size, hidden_size), device=my_device)
    b_hidden = torch.zeros((1, hidden_size), device=my_device)

    w_output = torch.randn(hidden_size, output_size, device=my_device)
    b_output = torch.zeros((1,output_size), device=my_device)

    lr = 0.005
    batch_size = 100
    eval_batch_size = 500 # used for validation or testing
    epochs = 500
    epochs = tqdm(range(epochs))

    data, labels = load_mnist(train=True, pin_memory=True)
    data_test, labels_test = load_mnist(train=False, pin_memory=True)

    for _ in epochs:
        w_hidden, b_hidden, w_output, b_output = train_epoch(data, labels, w_hidden, b_hidden, w_output, b_output,lr, batch_size)
        accuracy = evaluate(data_test, labels_test, w_hidden, b_hidden, w_output, b_output, eval_batch_size)
        epochs.set_postfix_str(f"accuracy = {accuracy}")

       # accuracy = evaluate(data_test, labels_test, w, b, eval_batch_size)
        #epochs.set_postfix_str(f"accuracy = {accuracy}")



if __name__ == '__main__':
    train()