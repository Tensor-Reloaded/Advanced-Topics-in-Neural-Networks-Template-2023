import torch
from torch import Tensor
from torch.utils.data import TensorDataset
from perceptron import Perceptron

class PerceptronLayer:
    def __init__(self, weights: Tensor, biases: Tensor, learningRate: float):
        self.perceptrons = [
            Perceptron(
                w, 
                b, 
                learningRate
            ) for w, b in zip(weights, biases)
        ]

    def forward(self, x: Tensor) -> Tensor:
        outputs = [p.forward(x) for p in self.perceptrons]
        predictedY = torch.tensor(outputs)
        return predictedY

    def backward(self, x: Tensor, observedY: Tensor, predictedY: Tensor):
        for p, oy, py in zip(self.perceptrons, observedY, predictedY):
            p.backward(x, oy, py)

    def train(self, trainingData: TensorDataset, epochs: int = 1):
        for epoch in range(epochs):
            for x, observedY in trainingData:
                predictedY = self.forward(x)
                self.backward(x, observedY, predictedY)

    def getAccuracy(self, dataset: TensorDataset) -> float:
        correct_predictions = 0
        for x, observedY in dataset:
            predictedY = self.forward(x)
            if torch.argmax(predictedY) == torch.argmax(observedY):
                correct_predictions += 1

        return correct_predictions / len(dataset)

