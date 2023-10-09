import torch
from multiclass_logistic_regression import MulticlassLogisticRegression
from torch.utils.data import TensorDataset

def main():
    # Given data
    x = torch.tensor([1, 3, 0], dtype=torch.float)
    y = torch.tensor([0, 1, 0], dtype=torch.float)

    # Initialize Neuron
    W = torch.tensor([[0.3, 0.1, -2], [-0.6, -0.5, 2], [-1, -0.5, 0.1]], dtype=torch.float)
    b = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float)
    eta = 0.1
    neuron = MulticlassLogisticRegression(W, b, eta)

    # Train the Neuron on the given data
    print("Initial weights:", neuron.W)
    print("Initial biases:", neuron.b, "\n")
    print("Initial prediction:", neuron.forward(x), "(Real observation:", y, ")", "\n\n")

    training_data = TensorDataset(x.unsqueeze(0), y.unsqueeze(0))
    neuron.train(training_data)

    print("Updated weights:", neuron.W)
    print("Updated biases:", neuron.b, "\n")
    print("Updated prediction:", neuron.forward(x), "(Real observation:", y, ")", "\n\n")

    # I chose to do it more times to see if the predicted y_hat indeed approaches the observed y
    epochs = 1000
    neuron.train(training_data, epochs)

    print("Updated(x",epochs,") weights:", neuron.W)
    print("Updated(x",epochs,") biases:", neuron.b, "\n")
    print("Updated(x",epochs,") prediction:", neuron.forward(x), "(Real observation:", y, ")", "\n\n")

if __name__ == "__main__":
    main()