import gzip
import pickle
import time

import torch
from tqdm import tqdm

from Activations import Activations
from Layer import Layer


class NeuralNetwork:
    LEARNING_RATE = 0.5
    REGULARIZATION_RATE = 0.001

    def __init__(self):
        self.testing_set = None
        self.validation_set = None
        self.training_set = None
        self.layers = []

    def load_input(self):
        def _map_data(given_set):
            data = given_set[0]
            tags = given_set[1]
            output = []

            for index in range(len(tags)):
                output += [(torch.from_numpy(data[index]).view(784, 1), tags[index])]

            return output

        with gzip.open("mnist.pkl.gz", "rb") as fd:
            training_set, validation_set, testing_set = pickle.load(fd, encoding='latin')

        self.training_set = _map_data(training_set)
        self.validation_set = _map_data(validation_set)
        self.testing_set = _map_data(testing_set)

    def add_layer(self, input_size, output_size, activation_callback):
        self.layers.append(Layer(input_size, output_size, activation_callback))

    def nn_feedforward(self, input_values):
        self.layers[0].feedforward(input_values)
        self.layers[1].feedforward(self.layers[0].activated_output)

    def nn_backpropagate(self, expected_result):
        weights_update = [
            torch.zeros(layer.weights.shape) for layer in self.layers
        ]
        biases_update = [
            torch.zeros(layer.biases.shape) for layer in self.layers
        ]

        delta = self.layers[1].activated_output - expected_result.unsqueeze(-1)
        biases_update[1] = delta
        weights_update[1] = torch.matmul(delta, self.layers[0].activated_output.T)

        delta = torch.matmul(self.layers[1].weights.t(), delta) * Activations.sigmoid_derivative(self.layers[0].output)
        biases_update[0] = delta
        weights_update[0] = torch.matmul(delta, self.layers[0].input.T)

        return weights_update, biases_update

    def train_mini_batch(self, data_set, max_iterations=10, batch_size=10, learning_rate=0.01):

        batch_count = len(data_set) // batch_size

        for it in range(max_iterations):

            for i in tqdm(range(batch_count), unit=" mini batches", desc=f"Epoch {it + 1} / {max_iterations}"):

                weights_adjustments = [
                    torch.zeros(layer.weights.shape) for layer in self.layers
                ]
                biases_adjustments = [
                    torch.zeros(layer.biases.shape) for layer in self.layers
                ]
                batch = data_set[i * batch_size: (i + 1) * batch_size]

                for input_values, target in batch:
                    expected_result = torch.zeros(10, dtype=torch.float)
                    expected_result[target] = 1

                    self.nn_feedforward(input_values)

                    weights_update, biases_update = self.nn_backpropagate(expected_result)
                    for params_idx in range(len(weights_adjustments)):
                        weights_adjustments[params_idx] += weights_update[params_idx]
                        biases_adjustments[params_idx] += biases_update[params_idx]

                for idx, layer in enumerate(self.layers):
                    layer.weights -= learning_rate * weights_adjustments[idx] / batch_size
                    layer.biases -= learning_rate * biases_adjustments[idx] / batch_size

            # self.test_model(self.testing_set)

    def predict(self, input_values):
        self.nn_feedforward(input_values.reshape(784, 1))
        return self.layers[-1].activated_output

    def test_model(self, test_set):
        wrong_predictions = 0
        correct_predictions = 0
        total_loss = 0.0

        for input_values, correct_tag in test_set:
            predicted = self.predict(input_values)
            predicted_value = torch.argmax(predicted)
            if predicted_value == correct_tag:
                correct_predictions += 1
            else:
                wrong_predictions += 1

            expected_result = torch.zeros(10, dtype=torch.float)
            expected_result[correct_tag] = 1
            total_loss += torch.nn.functional.cross_entropy(predicted.t(), expected_result.unsqueeze(-1).t())

        print(f"Correct: {correct_predictions}, "
              f"Wrong: {wrong_predictions}, "
              f"Total: {correct_predictions + wrong_predictions}, "
              f"Accuracy: {int(correct_predictions / (correct_predictions + wrong_predictions) * 10000.) / 100}%, "
              f"Average Loss: {total_loss / len(test_set)}\n")

        time.sleep(1)


if __name__ == '__main__':
    model = NeuralNetwork()
    model.load_input()

    model.add_layer(784, 100, Activations.sigmoid)
    model.add_layer(100, 10, Activations.softmax)

    model.train_mini_batch(model.training_set, 10, 32, 0.5)

    print('Testing set:')
    model.test_model(model.testing_set)
    # Testing set:
    # Correct: 9745, Wrong: 255, Total: 10000, Accuracy: 97.45 %, Average Loss: 1.4963749647140503

    print('Validation set:')
    model.test_model(model.validation_set)
    # Validation set:
    # Correct: 9751, Wrong: 249, Total: 10000, Accuracy: 97.51 %, Average Loss: 1.4948376417160034
