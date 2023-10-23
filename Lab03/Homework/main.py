import math
import random
from typing import Tuple
import numpy as np
import torch
from torch import Tensor
from torchvision.datasets import MNIST
import torch.utils.data
from tqdm import tqdm
import matplotlib.pyplot as plt


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


def to_one_hot(x: Tensor, device: torch.device = get_default_device()) -> Tensor:
    return torch.eye(x.max() + 1).to(device)[x]


def construct_dataset(dataset, for_training: bool = False, pin_memory: bool = True,
                      device: torch.device = get_default_device()):
    data = []
    labels = []
    for image, label in dataset:
        tensor = torch.from_numpy(np.array(image))
        data.append(tensor.to(device))
        labels.append(label)

    data = collate(data).float()  # shape 60000, 28, 28
    data = data.flatten(start_dim=1)  # shape 60000, 784

    data /= data.max()  # min max normalize

    labels = collate(labels).to(device)  # shape 60000
    if for_training:
        labels = to_one_hot(labels, device)  # shape 60000, 10
    if pin_memory:
        return data.pin_memory(), labels.pin_memory()
    return data, labels


def load_mnist(path: str = "./data", pin_memory: bool = True, device: torch.device = get_default_device()):
    test_set = MNIST(path, download=True, train=False)

    # TODO:Better way to split in a balanced way?
    #  Try using sklearn
    train_set, validation_set = torch.utils.data.random_split(MNIST(path, download=True, train=True), [50000, 10000],
                                                              torch.Generator().manual_seed(1))
    return construct_dataset(train_set, True, pin_memory, device), \
           construct_dataset(validation_set, False, pin_memory, device), \
           construct_dataset(test_set, False, pin_memory, device)


def sigmoid(x: Tensor) -> Tensor:
    return 1 / (1 + torch.exp(-x))


def feed_forward(x_input: Tensor, weights: list[Tensor], bias: list[Tensor], all_outputs: bool):
    no_layers = len(weights)
    outputs = [x_input]
    for layer in range(1, no_layers - 1):
        z = outputs[layer - 1] @ weights[layer] + bias[layer]
        outputs.append(sigmoid(z))

    z = outputs[no_layers - 2] @ weights[no_layers - 1] + bias[no_layers - 1]
    outputs.append(z.softmax(dim=1))

    return outputs if all_outputs else outputs[no_layers - 1]


# Uses cross entropy,softmax and the sigmoid function
def stochastic_gradient_descent(no_neurons_per_layer: list[int], device: torch.device, no_epochs: int, batch_size: int,
                                learning_rate: float,
                                weight_decay: float,
                                training_set: Tuple[Tensor, Tensor], validation_set: Tuple[Tensor, Tensor],
                                compute_per_epoch: bool) -> dict:
    no_layers = len(no_neurons_per_layer)

    training_loss = []
    validation_loss = []
    training_accuracy = []
    validation_accuracy = []

    weights = [[]]
    bias = [[]]
    for layer in range(1, no_layers):
        standard_deviation = 1 / (math.sqrt(no_neurons_per_layer[layer - 1]))
        mean = 0

        weights.append(torch.normal(mean, standard_deviation,
                                    size=(no_neurons_per_layer[layer - 1], no_neurons_per_layer[layer])).to(device))
        bias.append(torch.normal(mean, standard_deviation,
                                 size=(1, no_neurons_per_layer[layer])).to(device))

    no_instances = training_set[0].shape[0]

    # Used to shuffle the instances in each epoch in order to converge faster
    indexes = [*range(no_instances)]

    for epoch in range(no_epochs):
        random.shuffle(indexes)

        for start in range(0, no_instances, batch_size):
            # Add the first input as the output of the first layer
            end = min(no_instances, start + batch_size)

            target = training_set[1][indexes[start:end]]

            # Feed Forward
            output = feed_forward(training_set[0][indexes[start:end]], weights, bias, True)

            previous_layer_error = output[no_layers - 1] - target
            next_layer_error = None

            for layer in range(no_layers - 1, 0, -1):
                # compute the next layer error,after which we will modify the weights and biases
                if layer > 1:
                    next_layer_error = output[layer - 1] * (1 - output[layer - 1]) * (previous_layer_error @ weights[
                        layer].T)

                weights[layer] = (1 - learning_rate * weight_decay / no_instances) * weights[layer] - learning_rate * (
                        output[layer - 1].T @ previous_layer_error) / (end - start)
                bias += -learning_rate * (previous_layer_error.sum(dim=0))
                previous_layer_error = next_layer_error

        if compute_per_epoch:
            training_accuracy.append(compute_accuracy(training_set, weights, bias))
            validation_accuracy.append(compute_accuracy(validation_set, weights, bias))

            # training_loss.append(
            #     torch.nn.functional.cross_entropy(feed_forward(training_set[0], weights, bias, all_outputs=False),
            #                                       training_set[1]))
            # validation_loss.append(
            #     torch.nn.functional.cross_entropy(feed_forward(validation_loss[0], weights, bias, all_outputs=False),
            #                                       validation_loss[1]))
    dictionary = dict()
    dictionary["no_epochs"] = no_epochs
    dictionary["weights"] = weights
    dictionary["bias"] = bias
    if compute_per_epoch:
        dictionary["training_accuracy"] = training_accuracy
        dictionary["validation_accuracy"] = validation_accuracy
        dictionary["training_loss"] = training_loss
        dictionary["validation_loss"] = validation_loss

    return dictionary


def compute_accuracy(dataset: Tuple[Tensor, Tensor], weights: list[Tensor], bias: list[Tensor],
                     batch_size: int = 128) -> float:
    # Labels are not one hot encoded, because we do not need them as one hot.
    data, labels = dataset
    total_correct_predictions = 0
    no_instances = data.shape[0]
    non_blocking = weights[1].device.type == 'cuda'
    for i in range(0, no_instances, batch_size):
        x = data[i: i + batch_size].to(weights[1].device, non_blocking=non_blocking)
        y = labels[i: i + batch_size].to(weights[1].device, non_blocking=non_blocking)

        # If the target is one hot encoded
        if len(y.shape) > 1:
            target_max_value, y = torch.max(y, dim=1)

        predicted_distribution = feed_forward(x, weights, bias, False)

        predicted_max_value, predicted_max_value_indices = torch.max(predicted_distribution, dim=1)

        equality_mask = predicted_max_value_indices == y

        correct_predictions = equality_mask.sum().item()

        total_correct_predictions += correct_predictions

    return total_correct_predictions / no_instances


def main():
    device = get_default_device()
    print(f"Using device {device}")
    pin_memory = False

    training_set, validation_set, test_set = load_mnist(pin_memory=pin_memory, device=device)

    no_neurons_per_layer = [784, 100, 10]
    no_epochs = 30
    batch_size = 128
    learning_rate = 0.1
    weight_decay = 1
    compute_per_epoch = True

    model_data = stochastic_gradient_descent(no_neurons_per_layer, device, no_epochs, batch_size, learning_rate,
                                             weight_decay,
                                             training_set, validation_set, compute_per_epoch)

    if compute_per_epoch:
        x_points = [*range(1, model_data["no_epochs"] + 1)]

        plt.plot(x_points, model_data["training_accuracy"], label="Training")
        plt.title("Accuracy over no of epochs")
        plt.xlabel("No of Epochs")
        plt.ylabel("Accuracy")

        plt.plot(x_points, model_data["validation_accuracy"], label="Validation")
        plt.legend()
        plt.show()

    weights = model_data["weights"]
    bias = model_data["bias"]

    print("Training set accuracy: ", compute_accuracy(training_set, weights, bias, batch_size))
    print("Validation set accuracy: ", compute_accuracy(validation_set, weights, bias, batch_size))
    print("Test set accuracy: ", compute_accuracy(test_set, weights, bias, batch_size))

# Training set accuracy:  0.96172
# Validation set accuracy:  0.9537
# Test set accuracy:  0.9564


if __name__ == '__main__':
    main()
