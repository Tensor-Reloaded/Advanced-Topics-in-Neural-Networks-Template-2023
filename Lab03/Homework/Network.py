import gzip
import torch
import pickle
import Activations
import torch.nn as nn
from Layer import Layer
from typing import Callable


class Network:
    LEARNING_RATE = 0.5
    BATCH_SIZE = 30
    ITERATIONS = 20

    def __init__(self, dataset_file_path: str):
        self.training_set = None
        self.validation_set = None
        self.testing_set = None

        self.layers = []
        self.load_dataset(dataset_file_path)

    def load_dataset(self, dataset_file_path: str):
        with gzip.open(dataset_file_path, "rb") as fd:
            training_set, validation_set, testing_set = pickle.load(fd, encoding='latin')

        self.training_set = [(torch.tensor(inputs, dtype=torch.float32), tag) for inputs, tag in zip(training_set[0], training_set[1])]
        self.validation_set = [(torch.tensor(inputs, dtype=torch.float32), tag) for inputs, tag in zip(validation_set[0], validation_set[1])]
        self.testing_set = [(torch.tensor(inputs, dtype=torch.float32), tag) for inputs, tag in zip(testing_set[0], testing_set[1])]

        print("Successfully loaded dataset!")

    def add_layer(self, input_size: int, output_size: int, activation_callback: Callable):
        self.layers.append(Layer(input_size, output_size, activation_callback))

    def feedforward(self, input_data):
        self.layers[0].feedforward(input_data)
        self.layers[1].feedforward(self.layers[0].activated_output)

    def backpropagate(self, expected_result):
        weights_update = [torch.zeros(layer.weights.shape) for layer in self.layers]
        biases_update = [torch.zeros(layer.biases.shape) for layer in self.layers]

        delta = self.layers[1].activated_output - expected_result.squeeze()
        biases_update[1] = delta
        weights_update[1] = torch.matmul(delta.unsqueeze(-1), self.layers[0].activated_output.t().unsqueeze(0))

        delta = torch.matmul(self.layers[1].weights.t(), delta) * Activations.sigmoid_derivative(self.layers[0].output)
        biases_update[0] = delta
        weights_update[0] = torch.matmul(delta.unsqueeze(-1), self.layers[0].input.t().unsqueeze(0))

        return weights_update, biases_update

    def train(self, training_set, testing_set):
        batch_count = len(training_set) // self.BATCH_SIZE

        for iteration_idx in range(self.ITERATIONS):
            print(f"[TRAINING] Running iteration number {iteration_idx + 1}...")

            for batch_idx in range(batch_count):
                batch_start_idx = batch_idx * self.BATCH_SIZE
                batch_end_idx = (batch_idx + 1) * self.BATCH_SIZE
            
                batch_weights_update = [torch.zeros(layer.weights.shape) for layer in self.layers]
                batch_biases_update = [torch.zeros(layer.biases.shape) for layer in self.layers]

                input_data = training_set[batch_start_idx:batch_end_idx]

                for input_values, target in input_data:
                    expected_result = torch.Tensor([1 if i == target else 0 for i in range(10)]).reshape(10, 1)

                    self.feedforward(input_values)
                    weights_update, biases_update = self.backpropagate(expected_result)
                    for params_idx in range(len(batch_weights_update)):
                        batch_weights_update[params_idx] += weights_update[params_idx]
                        batch_biases_update[params_idx] += biases_update[params_idx]

                for idx, layer in enumerate(self.layers):
                    layer.weights -= self.LEARNING_RATE * batch_weights_update[idx] / self.BATCH_SIZE
                    layer.biases -= self.LEARNING_RATE * batch_biases_update[idx] / self.BATCH_SIZE

            if testing_set:
                self.test_accuracy(testing_set)

    def predict(self, input_values):
        self.feedforward(input_values)
        return self.layers[-1].activated_output
    
    def test_accuracy(self, test_set):
        correct = 0
        total_loss = 0

        for input_values, target_label in test_set:
            activated_output = self.predict(input_values)

            target_label_arr = torch.zeros(10, dtype=torch.float)
            target_label_arr[target_label] = 1

            total_loss += torch.nn.functional.cross_entropy(activated_output.unsqueeze(0), target_label_arr.unsqueeze(0))
            if torch.argmax(activated_output) == target_label:
                correct += 1
        
        acc_percentage = correct / len(test_set) * 100
        print(f"Accuracy: {correct}/{len(test_set)} ({round(acc_percentage, 2)}%) | Loss: {total_loss / len(test_set)}\n")