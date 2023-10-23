import torch
import time

from torch import Tensor
from torch.utils.data import DataLoader
from typing import Callable

from timed import timedCuda

class ThreeLayerModel:
    def __init__(self, inputLayerSize, hiddenLayerSize, outputLayerSize, device):

        self.inputLayerSize = inputLayerSize
        self.hiddenLayerSize = hiddenLayerSize
        self.outputLayerSize = outputLayerSize

        self.device = device

        self.hiddenLayer = Layer(inputs=inputLayerSize, neurons=hiddenLayerSize, activation=torch.sigmoid, activationDerivative=lambda x: torch.sigmoid(x) * (1 - torch.sigmoid(x)), device=device)
        self.outputLayer = Layer(inputs=hiddenLayerSize, neurons=outputLayerSize, activation=lambda x : torch.softmax(x, dim=1), device=device)
   
    def forward(self, featureBatch: Tensor) -> Tensor:

        hiddenLabels = self.hiddenLayer.forward(featureBatch)

        outputLabels = self.outputLayer.forward(hiddenLabels)

        return outputLabels
    
    def backward(self, featureBatch: Tensor, observedLabelsBatch: Tensor, learningRate: float) -> float:

        hiddenLabelsBatch = self.hiddenLayer.forward(featureBatch)
        predictedLabelsBatch = self.outputLayer.forward(hiddenLabelsBatch)

        outputError = predictedLabelsBatch - observedLabelsBatch

        hiddenError = torch.matmul(outputError, self.outputLayer.weights.t()) * self.hiddenLayer.activationDerivative(hiddenLabelsBatch)


        self.outputLayer.backward(hiddenLabelsBatch, outputError, learningRate)
        self.hiddenLayer.backward(featureBatch, hiddenError, learningRate)

        return torch.nn.functional.cross_entropy(predictedLabelsBatch, observedLabelsBatch).item()

    def train(self, dataloader: DataLoader, epochs: int, startingLeagningRate: float, learningRateDecayPercentage: float):

        print(f"Starting training...")
        startTime = time.time()
        totalLoss = 0.0

        currentLearningRate = startingLeagningRate

        for epoch in range(epochs):
            for featuresBatch, observedLabelsBatch in dataloader:

                featuresBatch = featuresBatch.to(self.device)
                observedLabelsBatch = observedLabelsBatch.to(self.device)
                loss = self.backward(featuresBatch, observedLabelsBatch, currentLearningRate)

            currentLearningRate = currentLearningRate - currentLearningRate * learningRateDecayPercentage

            epochEndTime = time.time()
            epochElapsedTime = epochEndTime - startTime
            totalLoss += loss
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss}, Time elapsed: {epochElapsedTime} seconds")

        endTime = time.time()
        elapsedTime = endTime - startTime
        print(f"Training ended, Total time: {elapsedTime} seconds")
    
    def getAccuracy(self, dataloader: DataLoader) -> float:

        correct_predictions = 0
        total_instances = 0

        for featuresBatch, observedLabelsBatch in dataloader:

            featuresBatch = featuresBatch.to(self.device)
            observedLabelsBatch = observedLabelsBatch.to(self.device)

            predictedY = self.forward(featuresBatch)
            correct_predictions += (torch.argmax(predictedY, dim=1) == torch.argmax(observedLabelsBatch, dim=1)).sum().item()
            total_instances += featuresBatch.size(0)
            
        return correct_predictions / total_instances

class Layer:
    def __init__(self, inputs: int, neurons: int, activation: Callable[[Tensor],Tensor], activationDerivative: Callable[[Tensor],Tensor]=lambda x:x, device: torch.device=torch.device('cpu')):

        self.inputs = inputs
        self.neurons = neurons
        self.activation = activation
        self.activationDerivative = activationDerivative

        self.weights = (1 + 1) * torch.rand(self.inputs, self.neurons, device=device) - 1
        self.biases = torch.rand(self.neurons, device=device)

    def forward(self, featureBatch: Tensor) -> Tensor:

        z = torch.matmul(featureBatch, self.weights) + self.biases

        labels = self.activation(z)
        return labels
    
    def backward(self, inputLabelsBatch: Tensor, error: Tensor, learningRate: float):

        gradient = inputLabelsBatch.t() @ error

        self.weights.sub_(learningRate * gradient)

        self.biases.sub_(learningRate * error.mean(dim=0))
