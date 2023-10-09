import torch
from perceptron import Perceptron

def main():
    # Given data
    x = torch.tensor([1, 3, 0], dtype=torch.float)
    y = torch.tensor([0, 1, 0], dtype=torch.float)

    # Initialize Perceptron
    W = torch.tensor([[0.3, 0.1, -2], [-0.6, -0.5, 2], [-1, -0.5, 0.1]], dtype=torch.float)
    b = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float)
    eta = 0.1
    perceptron = Perceptron(W, b, eta)

    # Train the Perceptron on the given data
    print("Initial weights:", perceptron.W)
    print("Initial biases:", perceptron.b, "\n")
    print("Initial prediction:", perceptron.forward(x), "(Real observation:", y, ")", "\n\n")

    perceptron.train_step(x,y)

    print("Updated weights:", perceptron.W)
    print("Updated biases:", perceptron.b, "\n")
    print("Updated prediction:", perceptron.forward(x), "(Real observation:", y, ")", "\n\n")

    # I chose to do it more times to see if the predicted y_hat indeed approaches the observed y
    epochs = 1000
    for epoch in range(epochs):
        perceptron.train_step(x,y)

    print("Updated(x",epochs,") weights:", perceptron.W)
    print("Updated(x",epochs,") biases:", perceptron.b, "\n")
    print("Updated(x",epochs,") prediction:", perceptron.forward(x), "(Real observation:", y, ")", "\n\n")

if __name__ == "__main__":
    main()