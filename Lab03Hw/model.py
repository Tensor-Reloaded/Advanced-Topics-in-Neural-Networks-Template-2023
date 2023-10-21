import torch
from torch import Tensor
from torch.utils.data import DataLoader
from typing import Callable

class ThreeLayerModel:
    def __init__(self, inputLayerSize, hiddenLayerSize, outputLayerSize):
        self.hiddenLayer = Layer(inputs=inputLayerSize, neurons=hiddenLayerSize, activation=torch.sigmoid)
        self.outputLayer = Layer(inputs=hiddenLayerSize, neurons=outputLayerSize, activation=lambda x : torch.softmax(x, dim=1))
        pass
    
    def forward(self, featureBatch: Tensor) -> Tensor:

        assert featureBatch.dim() == 2, f"[{ThreeLayerModel.__name__}] Expected featureBatch to have 2 dimensions, but found ({featureBatch.dim()})"
        assert featureBatch.size(1) == self.hiddenLayer.inputs, f"[{ThreeLayerModel.__name__}] Expected features to be of size ({self.hiddenLayer.inputs}), but found ({featureBatch.size(1)})"

        hiddenLabels = self.hiddenLayer.forward(featureBatch)

        outputLabels = self.outputLayer.forward(hiddenLabels)

        return outputLabels
    
    def getAccuracy(self, dataloader: DataLoader) -> float:

        correct_predictions = 0
        total_instances = 0

        for batch in dataloader:
            x, observedY = batch
            predictedY = self.forward(x)
            correct_predictions += (torch.argmax(predictedY, dim=1) == torch.argmax(observedY, dim=1)).sum().item()
            total_instances += x.size(0)
            
        return correct_predictions / total_instances

class Layer:
    def __init__(self, inputs: int, neurons: int, activation: Callable[[Tensor],Tensor]):

        self.__validateActivation(neurons, activation)

        self.inputs = inputs
        self.neurons = neurons
        self.activation = activation

        self.weights = torch.randn(self.inputs, self.neurons)
        self.biases = torch.randn(self.neurons)

    def __validateActivation(self, neurons, activation: Callable[[Tensor],Tensor]):

        assert callable(activation), f"[{Layer.__name__}] Expected activation function to be callable"

        test_input = torch.randn(2, neurons)
        test_output = activation(test_input)
        assert test_output.shape == test_input.shape, f"[{Layer.__name__}] Expected activation function to return output of same shape as input"

    def forward(self, featureBatch: Tensor) -> Tensor:

        assert featureBatch.dim() == 2, f"[{Layer.__name__}] Expected featureBatch to have 2 dimensions, but found ({featureBatch.dim()})"
        assert featureBatch.size(1) == self.inputs, f"[{Layer.__name__}] Expected features to be of size ({self.inputs}), but found ({featureBatch.size(1)})"

        z = torch.matmul(featureBatch, self.weights) + self.biases

        labels = self.activation(z)
        return labels
