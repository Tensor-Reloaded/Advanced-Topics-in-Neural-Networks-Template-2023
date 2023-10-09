import torch
import torch.nn as nn


def feed_forward(input, weights, biases):
    weights_t = torch.transpose(weights, 0, 1)
    z = torch.matmul(weights_t, input) + biases
    return nn.Softmax(dim=0)(z)


def compute_gradients(input, predicted_values, labels):
    gradient_w = torch.mul(predicted_values - labels, input)
    gradient_b = predicted_values - labels

    return gradient_w, gradient_b

def main():
    x = torch.tensor([1.0, 3.0, 0])
    biases = torch.tensor([0.1, 0.1, 0.1])
    learning_rate = 1

    weights = torch.tensor([
        [0.3, 0.1, -2],
        [-0.6, -0.5, 2],
        [-1, -0.5, 0.1]
    ])

    labels = torch.tensor([0, 1.0, 0])

    predicted_values = feed_forward(x, weights, biases)
    print(f"Predicted values: {predicted_values.tolist()}")

    delta_w, delta_b = compute_gradients(x, predicted_values, labels)
    print(f"Delta weights: {delta_w.tolist()}")
    print(f"Delta biases: {delta_b.tolist()}")

    weights -= delta_w * learning_rate
    biases -= delta_b * learning_rate

    print(f"Adjusted weights: {weights}")
    print(f"Adjusted biases: {biases}")


if __name__ == "__main__":
    main()