import torch
from torch import Tensor
from torch.utils.data import TensorDataset

class MulticlassLogisticRegression:
    def __init__(self, W: Tensor, b: Tensor, eta: float):
        self.W = W # weights tensor
        self.b = b # biases tensor
        self.eta = eta # learning rate

    def forward(self, x: Tensor) -> Tensor:
        z = torch.matmul(self.W.t(), x) + self.b
        y_hat = torch.softmax(z, dim=0)
        return y_hat

    def backward(self, x: Tensor, y: Tensor, y_hat: Tensor) -> tuple:
        grad_z_L = y_hat - y
        grad_W_L = grad_z_L.view(-1, 1) * x
        grad_b_L = grad_z_L
        return grad_W_L, grad_b_L

    def update(self, grad_W_L: Tensor, grad_b_L: Tensor):
        self.W -= self.eta * grad_W_L.t()
        self.b -= self.eta * grad_b_L

    def train(self, trainingData: TensorDataset, epochs: int = 1):
        for epoch in range(epochs):
            for x, y in trainingData:
                y_hat = self.forward(x)
                grad_W_L, grad_b_L = self.backward(x, y, y_hat)
                self.update(grad_W_L, grad_b_L)

    def getAccuracy(self, dataset: TensorDataset) -> float:
        correct_predictions = 0
        for x, y in dataset:
            y_hat = self.forward(x)
            if torch.argmax(y_hat) == torch.argmax(y):
                correct_predictions += 1

        return correct_predictions / len(dataset)
