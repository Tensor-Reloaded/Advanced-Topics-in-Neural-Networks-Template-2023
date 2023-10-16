import torch
from torch import Tensor
from torch.utils.data import TensorDataset

class Perceptron:
    def __init__(self, weights: Tensor, bias: float, learningRate: float):
        self.weights = weights
        self.bias = bias
        self.learningRate = learningRate

    def forward(self, x: Tensor) -> float:
        z = torch.matmul(self.weights, x) + self.bias
        predictedY = torch.sigmoid(z)
        return predictedY

    def backward(self, x: Tensor, observedY: float, predictedY: float):
        grad_z_L = predictedY - observedY
        grad_W_L = grad_z_L * x
        grad_b_L = grad_z_L

        self.weights -= self.learningRate * grad_W_L
        self.bias -= self.learningRate * grad_b_L

    def train(self, trainingData: TensorDataset, epochs: int = 1):
        for epoch in range(epochs):
            for x, observedY in trainingData:
                predictedY = self.forward(x)
                self.backward(x, observedY, predictedY)
