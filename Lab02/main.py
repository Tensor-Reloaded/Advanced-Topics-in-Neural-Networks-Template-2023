import math
import torch
from torchvision import datasets
import numpy as np
from tqdm.auto import tqdm
import time


def activation_function(z):
    return 1 / (1 + math.exp(-z))


def train_perceptron(x_inputs: torch.tensor, weights: torch.tensor, b: torch.tensor, y_true: torch.tensor, mu: float,
                     epochs: int):
    for e in range(epochs):
        print('Epoch:', e + 1)

        # The first for loop function is used in order to iterate through each training instance
        y_hat = torch.zeros([len(x_inputs)], dtype=int)

        for (instance_no, i_bar) in zip(range(len(x_inputs)), tqdm(range(len(x_inputs)))):
            target_class = y_true[instance_no]
            one_hot_encoded_label = torch.zeros([10], dtype=int)
            one_hot_encoded_label[target_class] = 1

            xi = x_inputs[instance_no] / 255  # Normalize function within range (0,1)

            # The second for loop is used to determine maximum value of dot product between weights and input values
            # which represents the prediction for a single training example for a group of 10 perceptrons
            z_values = torch.zeros([10])
            y_list = torch.zeros([10])

            for i in range(10):
                z = (torch.dot(xi.float(), weights[:, i].float()) + b[i])

                prediction = activation_function(z)

                z_values[i] = z
                y_list[i] = prediction  # Binary values (0 or 1)

            y_hat[instance_no] = torch.argmax(z_values)

            for w in range(10):
                weights[:, w] = weights[:, w] + (mu * (one_hot_encoded_label[w] - y_list[w]) * xi)
                b[w] = b[w] + mu * (one_hot_encoded_label[w] - y_list[w])

        # Determine the accuracy
        correct = 0
        for ind in range(len(y_true)):
            if y_hat[ind] == y_true[ind]:
                correct += 1

        accuracy = round(correct / float(len(y_true)) * 100, 2)
        print('Training accuracy:', accuracy, '%')
        time.sleep(0.01)

    # return model
    return (weights, b)


def test(x_inputs: torch.tensor, weights: torch.tensor, b: torch.tensor, y_true: torch.tensor):
    # The first for loop function is used in order to iterate through each test instance
    y_hat = torch.zeros([len(x_inputs)], dtype=int)
    for (instance_no, i_bar) in zip(range(len(x_inputs)), tqdm(range(len(x_inputs)))):
        target_class = y_true[instance_no]
        one_hot_encoded_label = torch.zeros([10], dtype=int)
        one_hot_encoded_label[target_class] = 1

        xi = x_inputs[instance_no] / 255  # Normalize function within range (0,1)

        # The second for loop is used for maximum value of dot product between weights and input values which
        # represents the prediction for a single test instance for a group of 10 perceptrons
        z_values = torch.zeros([10])
        y_list = torch.zeros([10])

        for i in range(10):
            z = (torch.dot(xi, weights[:, i]) + b[i])
            prediction = activation_function(z)

            z_values[i] = z
            y_list[i] = prediction  # Binary values (0 or 1)

        y_hat[instance_no] = torch.argmax(z_values)

    # Determine the accuracy
    correct = 0
    for ind in range(len(y_true)):
        if y_hat[ind] == y_true[ind]:
            correct += 1

    accuracy = round(correct / float(len(y_true)) * 100, 2)
    print('Test accuracy:', accuracy, '%')
    time.sleep(0.01)


if __name__ == '__main__':
    # Train Data
    train_data = datasets.MNIST(
        root='D:\Master_UAIC\Facultate_Discipline\Anul_1\Sem_I\Capitole_AvansateDeRetele_Neuronale\Laborator\Lab_2\Assignment_Lab_2\Perceptron\perceptron_data',
        train=True, download=True)

    # Test Data
    test_data = datasets.MNIST(
        root='D:\Master_UAIC\Facultate_Discipline\Anul_1\Sem_I\Capitole_AvansateDeRetele_Neuronale\Laborator\Lab_2\Assignment_Lab_2\Perceptron\perceptron_data',
        train=False, download=True)

    # ---------- Start train data ----------
    converted_images = []
    converted_labels = []

    for i in range(len(train_data)):
        converted_images.append((np.array(train_data[i][0])).flatten())
        converted_labels.append(train_data[i][1])

    x_inputs = torch.tensor(np.array(converted_images), dtype=float)

    # Initial Weights = tensor with a given size which contains random values between -0.05 and 0.05
    weights = torch.tensor(np.random.uniform(-0.05, 0.05, (784, 10)), dtype=float)  # 10 represents the number of trained perceptrons

    # Initial biases
    b = torch.tensor(np.random.uniform(-0.05, 0.05, 10), dtype=float)

    # Create the tensor which contains the true labels
    y_true = torch.tensor(converted_labels)

    # Define the learning rate
    mu = 0.01

    # ---------- End train data ----------

    # ---------- Start test data ----------
    test_converted_images = []
    test_converted_labels = []

    for i in range(len(test_data)):
        test_converted_images.append((np.array(test_data[i][0])).flatten())
        test_converted_labels.append(test_data[i][1])

    test_x_inputs = torch.tensor(np.array(test_converted_images), dtype=float)

    # Initial Weights = tensor with a given size which contains random values between -0.05 and 0.05
    test_weights = torch.tensor(
        np.random.uniform(-0.05, 0.05, (784, 10)), dtype=float)  # 10 represents the number of trained perceptrons

    # Initial biases
    test_b = torch.tensor(np.random.uniform(-0.05, 0.05, 10), dtype=float)

    # Create the tensor which contains the true labels
    test_y_true = torch.tensor(test_converted_labels)

    # ---------- End test data ----------

    # Accuracy on test dataset before training
    print('Accuracy on test dataset before training')
    test(test_x_inputs, test_weights, test_b, test_y_true)
    print()

    epochs = 7
    trained_model = train_perceptron(x_inputs, weights, b, y_true, mu, epochs)

    # Accuracy on test dataset after training
    print('\nAccuracy on test dataset after {0} epochs'.format(epochs))
    test(test_x_inputs, trained_model[0], trained_model[1], test_y_true)
