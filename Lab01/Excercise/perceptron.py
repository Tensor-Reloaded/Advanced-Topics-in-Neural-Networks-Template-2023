import torch
from torch import Tensor

class Perceptron:
    def __init__(self, W: Tensor, b: Tensor, eta: float):
        self.W = W # weights tensor
        self.b = b # biases tensor
        self.eta = eta # learning rate

    # Get a prediction for a given input
    def forward(self, x: Tensor) -> Tensor:

        # Compute linear combinations
        z = torch.matmul(self.W.t(), x) + self.b
        
        # Compute prediction
        y_hat = torch.softmax(z, dim=0)
        return y_hat

    # Get the gradients for weights and biases for a given input, real output, and predicted output
    def backward(self, x: Tensor, y: Tensor, y_hat: Tensor) -> tuple:
        grad_z_L = y_hat - y
        grad_W_L = grad_z_L.view(-1, 1) * x
        grad_b_L = grad_z_L
        return grad_W_L, grad_b_L

    # Update the weights and biases for given gradients
    def update(self, grad_W_L: Tensor, grad_b_L: Tensor):
        self.W -= self.eta * grad_W_L.t()
        self.b -= self.eta * grad_b_L

    def train_step(self, x: Tensor, y: Tensor):
        y_hat = self.forward(x)
        grad_W_L, grad_b_L = self.backward(x, y, y_hat)
        self.update(grad_W_L, grad_b_L)
