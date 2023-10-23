from typing import Tuple
import numpy as np
import torch
from torch import Tensor
from torchvision.datasets import MNIST
from tqdm import tqdm
import imutils
from adam_optimizer import AdamOptimizer


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def activate(x: Tensor) -> Tensor:
    return x.softmax(dim=1)


def hidden_activate(x: Tensor) -> Tensor:
    return torch.relu(x)


def deriv_activation(y: Tensor) -> Tensor:
    return torch.diag_embed(y) - y.unsqueeze(1).transpose(1, 2) @ y.unsqueeze(1)


def collate(x) -> Tensor:
    if isinstance(x, (tuple, list)):
        if isinstance(x[0], Tensor):
            return torch.stack(x)
        return torch.tensor(x)
    raise "Not supported yet"
    # see torch\utils\data\_utils\collate.py


def to_one_hot(x: Tensor) -> Tensor:
    return torch.eye(x.max() + 1)[x]


def forward(x: Tensor, w: Tensor, b: Tensor) \
        -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    z_h = x @ w[0] + b[0]
    y_h_hat = hidden_activate(z_h)  # activate over forward step
    std_mean = torch.std_mean(y_h_hat)
    y_h_hat -= std_mean[1]
    y_h_hat /= std_mean[0]
    z_h_1 = y_h_hat @ w[1] + b[1]
    y_h_hat_1 = hidden_activate(z_h_1)  # activate over forward step
    std_mean = torch.std_mean(y_h_hat_1)
    y_h_hat_1 -= std_mean[1]
    y_h_hat_1 /= std_mean[0]
    z = y_h_hat_1 @ w[2] + b[2]
    y_hat = activate(z)
    return z_h, y_h_hat, z, y_hat, z_h_1, y_h_hat_1


def train_batch(adam_optimizer0: AdamOptimizer, adam_optimizer1: AdamOptimizer, adam_optimizer2: AdamOptimizer,
                x: Tensor, y_true: Tensor, w: Tensor, b: Tensor, mu: float,
                batch_size: int, wd: float, step: int, epoch: int, nsteps: int) -> Tuple[Tensor, Tensor, Tensor]:
    # forward step
    z_h, y_h_hat, z, y_hat, z_h_1, y_h_hat_1 = forward(x, w, b)

    # backward step
    error = (y_hat - y_true)
    # w_2_copy = -2 * wd * w[2]
    # w_1_copy = -2 * wd * w[1]
    # w_0_copy = -2 * wd * w[0]
    loss = torch.nn.functional.cross_entropy(y_hat, y_true) + wd * torch.sum(torch.square(w[2]))
    error_h_1 = (error @ w[2].transpose(0, 1)) * (z_h_1.flatten() > 0).float().reshape(batch_size, -1)
    error_h = (error_h_1 @ w[1].transpose(0, 1)) * (z_h.flatten() > 0).float().reshape(batch_size, -1)
    # error_h = ((error @ w[1].transpose(0, 1)).unsqueeze(1)
    #            @ deriv_activation(y_h_hat)).squeeze(1)
    gradient_2 = (y_h_hat_1.transpose(0, 1) @ error.mean(axis=0)
                  .unsqueeze(0)
                  .repeat(batch_size, 1))
    gradient_1 = (y_h_hat.transpose(0, 1) @ error_h_1.mean(axis=0)
                  .unsqueeze(0)
                  .repeat(batch_size, 1))
    gradient_0 = (x.transpose(0, 1) @ error_h.mean(axis=0)
                  .unsqueeze(0)
                  .repeat(batch_size, 1))
    w_modif, b_modif = adam_optimizer2.update((epoch - 1) * nsteps + step, gradient_2, error.mean(axis=0))
    # w[2] += mu * w_2_copy
    w[2] -= mu * w_modif
    b[2] -= mu * b_modif
    w_modif, b_modif = adam_optimizer1.update((epoch - 1) * nsteps + step, gradient_1, error_h_1.mean(axis=0))
    # w[1] += mu * w_1_copy
    w[1] -= mu * w_modif
    b[1] -= mu * b_modif
    w_modif, b_modif = adam_optimizer0.update((epoch - 1) * nsteps + step, gradient_0, error_h.mean(axis=0))
    # w[0] += mu * w_0_copy
    w[0] -= mu * w_modif
    b[0] -= mu * b_modif
    return w, b, loss


def train_perceptron(adam_optimizer0: AdamOptimizer, adam_optimizer1: AdamOptimizer, adam_optimizer2: AdamOptimizer,
                     data: Tensor, labels: Tensor, w: Tensor, b: Tensor, mu: float,
                     batch_size: int, wd: float, epoch: int) -> Tuple[Tensor, Tensor, Tensor]:
    non_blocking = w[0].device.type == 'cuda'
    losses = []
    nsteps = data.shape[0] // batch_size
    for step in range(nsteps):
        #  select batch
        # batch = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
        batch = range(step * batch_size, (step + 1) * batch_size)
        input = data[batch].to(w[0].device, non_blocking=non_blocking)
        output = labels[batch].to(w[0].device, non_blocking=non_blocking)
        w, b, loss = train_batch(adam_optimizer0, adam_optimizer1, adam_optimizer2,
                                 input, output, w, b, mu, batch_size, wd, step + 1, epoch, nsteps)
        losses += [loss]
    return w, b, collate(losses).mean()


def load_dataset(path: str = "./data", train: bool = True, pin_memory: bool = True):
    mnist_trainset = MNIST(root=path, train=train, download=True)
    # 60.000 tuple (in, out)
    x_data = []
    y_data = []
    for image, label in mnist_trainset:
        img = np.array(image)
        tensor = torch.from_numpy(img)
        x_data.append(tensor)
        y_data.append(label)
        if train:
            # data augmentation
            img_shifted_1 = torch.cat((255 * torch.ones(size=(2, 28)), tensor[2:, :]))
            img_shifted_2 = torch.cat((tensor[:26, :], 255 * torch.ones(size=(2, 28))))
            img_shifted_3 = torch.cat((255 * torch.ones(size=(28, 2)), tensor[:, 2:]), dim=1)
            img_shifted_4 = torch.cat((tensor[:, :26], 255 * torch.ones(size=(28, 2))), dim=1)
            x_data += [img_shifted_1, img_shifted_2, img_shifted_3, img_shifted_4]
            y_data += [label] * 4

            img_rotated_1 = imutils.rotate(img, 5)
            img_rotated_2 = imutils.rotate(img, -5)
            x_data += [torch.from_numpy(img_rotated_1), torch.from_numpy(img_rotated_2)]
            y_data += [label] * 2
    x_data = collate(x_data).float()
    x_data = x_data.flatten(start_dim=1)  # shape 60000, 784
    maxi_data = x_data.max()
    x_data -= x_data.min()
    x_data /= maxi_data  # min max normalize
    mean_std = torch.std_mean(x_data)
    x_data -= mean_std[1]
    x_data /= mean_std[0]
    y_data = collate(y_data)  # shape 60000
    if train:
        y_data_labels = to_one_hot(y_data)  # shape 60000, 10
        if pin_memory:
            return x_data.pin_memory(), y_data.pin_memory(), y_data_labels.pin_memory()
        return x_data, y_data, y_data_labels
    if pin_memory:
        return x_data.pin_memory(), y_data.pin_memory()
    return x_data, y_data


def evaluate(data: Tensor, labels: Tensor, w: Tensor, b: Tensor,
             batch_size: int) -> Tuple[float, float]:
    # Labels are not one hot encoded, because we do not need them as one hot.
    total_correct_predictions = 0
    loss = []
    total_len = data.shape[0]
    non_blocking = w[0].device.type == 'cuda'
    for i in range(0, total_len, batch_size):
        x = data[i: i + batch_size].to(w[0].device, non_blocking=non_blocking)
        y = labels[i: i + batch_size].to(w[0].device, non_blocking=non_blocking)
        predicted_distribution = forward(x, w, b)
        loss += [torch.nn.functional.cross_entropy(predicted_distribution[3], y)]
        correct_predictions = (torch.max(predicted_distribution[3], dim=1)[1] == y).sum().item()
        total_correct_predictions += correct_predictions
    return total_correct_predictions / data.shape[0], collate(loss).mean()


def initialize_weights(device):
    stddev = np.sqrt(2 / (784 + 512))
    w1 = torch.normal(0, stddev, size=(784, 512), device=device)
    stddev = np.sqrt(2 / (512 + 512))
    w2 = torch.normal(0, stddev, size=(512, 512), device=device)
    stddev = np.sqrt(2 / (512 + 10))
    w3 = torch.normal(0, stddev, size=(512, 10), device=device)
    return [w1, w2, w3]


def initialize_biases(device):
    b1 = torch.rand((512,), device=device)
    b2 = torch.rand((512,), device=device)
    b3 = torch.rand((10,), device=device)
    return [b1, b2, b3]


def train(epochs: int = 1000, device: torch.device = get_default_device(),
          mu: float = 0.0005, batch_size: int = 100, eval_batch_size: int = 500,
          wd: float = 0.01):
    print(f"Using device {device}")
    pin_memory = device.type == 'cuda'  # Check the provided references.
    weights = initialize_weights(device)
    biases = initialize_biases(device)
    x_train, y_train_labels, y_train = load_dataset(train=True, pin_memory=pin_memory)
    x_test, y_test = load_dataset(train=False, pin_memory=pin_memory)
    # accuracy_test = [evaluate(x_test, y_test, weights, biases, eval_batch_size) / x_test.shape[0]]
    accuracies = []
    epochs = tqdm(range(epochs))
    adam_optimizer0 = AdamOptimizer()
    adam_optimizer1 = AdamOptimizer()
    adam_optimizer2 = AdamOptimizer()
    for epoch in epochs:
        if not (epoch + 1) % 60:
            mu *= 0.2
        weights, biases, loss = train_perceptron(adam_optimizer0, adam_optimizer1, adam_optimizer2,
                                                 x_train, y_train, weights, biases,
                                                 mu, batch_size, wd, epoch + 1)
        accuracy_test, loss_test = evaluate(x_test, y_test, weights, biases, eval_batch_size)
        accuracy_train, loss_train = evaluate(x_train, y_train_labels, weights, biases,
                                              eval_batch_size)
        epochs.set_postfix_str(f"accuracy_test = {accuracy_test}, loss_test = {loss_test}, "
                               f"accuracy_train = {accuracy_train}, loss_train = {loss_train}")
    print(max(accuracies))


if __name__ == '__main__':
    # adam optimizer
    train(200, mu=0.1, batch_size=60)
