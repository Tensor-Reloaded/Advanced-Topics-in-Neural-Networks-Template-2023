import torch
import torch.nn as nn

LEARNING_RATE = 1


def nn_feedforward(x, weights, biases):
    z = torch.matmul(torch.transpose(weights, 0, 1), x) + biases
    return nn.Softmax(dim=0)(z)


def compute_gradients(x, expected_output, predicted):
    return torch.matmul(x.view(-1, 1), (predicted - expected_output).view(1, -1)), predicted - expected_output


if __name__ == '__main__':
    x = torch.tensor([1.0, 3, 0])
    expected_output = torch.tensor([0.0, 1, 0])

    weights = torch.tensor([[0.3, 0.1, -2], [-0.6, -0.5, 2], [-1, -0.5, 0.1]])
    biases = torch.tensor([0.1, 0.1, 0.1])

    predicted_values = nn_feedforward(x, weights, biases)
    print(f'Predicted values:\n{predicted_values}\n')

    delta_weights, delta_biases = compute_gradients(x, expected_output, predicted_values)
    print(f'Delta weights:\n{delta_weights}\nDelta biases:\n{delta_biases}\n')

    weights = weights - LEARNING_RATE * delta_weights
    biases = biases - LEARNING_RATE * delta_biases

    print(f'Adjusted weights:\n{weights}\nAdjusted biases:\n{biases}\n')
