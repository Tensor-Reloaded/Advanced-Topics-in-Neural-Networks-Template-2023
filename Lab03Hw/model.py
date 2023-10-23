import torch
import time

from torch import Tensor
from torch.utils.data import DataLoader
from typing import Callable

# class Validator:
#     def __init__(self, class_):

#         self.class_ = class_

#     def validateTensorShapeAndSize(self, tensor: Tensor, tensorName:str, expectedFeaturesCount: int):

#         assert tensor.dim() == 2, f"[{self.class_.__name__}] Expected {tensorName} to have 2 dimensions, but found ({tensor.dim()})"
#         assert tensor.size(1) == expectedFeaturesCount, f"[{self.class_.__name__}] Expected {tensorName} to be of size ({expectedFeaturesCount}), but found ({tensor.size(1)})"

#     def validateActivation(self, activation: Callable[[Tensor],Tensor], numberOfNeurons: int):

#         assert callable(activation), f"[{Layer.__name__}] Expected activation function to be callable"

#         test_input = torch.randn(2, numberOfNeurons)
#         test_output = activation(test_input)
#         assert test_output.shape == test_input.shape, f"[{Layer.__name__}] Expected activation function to return output of same shape as input"

class ThreeLayerModel:
    def __init__(self, inputLayerSize, hiddenLayerSize, outputLayerSize, device):

        self.inputLayerSize = inputLayerSize
        self.hiddenLayerSize = hiddenLayerSize
        self.outputLayerSize = outputLayerSize

        self.device = device

        self.hiddenLayer = Layer(inputs=inputLayerSize, neurons=hiddenLayerSize, activation=torch.sigmoid, device=device)
        self.outputLayer = Layer(inputs=hiddenLayerSize, neurons=outputLayerSize, activation=lambda x : torch.softmax(x, dim=1), device=device)

        # self.__validator = Validator(ThreeLayerModel)
    
    def forward(self, featureBatch: Tensor) -> Tensor:

        # self.__validator.validateTensorShapeAndSize(featureBatch, "featureBatch", self.inputLayerSize)

        hiddenLabels = self.hiddenLayer.forward(featureBatch)

        outputLabels = self.outputLayer.forward(hiddenLabels)

        return outputLabels
    
    def backward(self, featureBatch: Tensor, observedLabelsBatch: Tensor, learningRate: float):

        # self.__validator.validateTensorShapeAndSize(featureBatch, "featureBatch", self.inputLayerSize)
        # self.__validator.validateTensorShapeAndSize(observedLabelsBatch, "observedLabelsBatch", self.outputLayerSize)


        hiddenLabelsBatch = self.hiddenLayer.forward(featureBatch)
        predictedLabelsBatch = self.outputLayer.forward(hiddenLabelsBatch)

        outputError = predictedLabelsBatch - observedLabelsBatch
        hiddenError = torch.matmul(outputError, self.outputLayer.weights.t())

        self.outputLayer.backward(hiddenLabelsBatch, outputError, learningRate)
        self.hiddenLayer.backward(featureBatch, hiddenError, learningRate)

    def train(self, dataloader: DataLoader, epochs: int, startingLeagningRate: float, learningRateDecayPercentage: float):

        print(f"Starting training...")
        startTime = time.time()
        totalLoss = 0.0

        currentLearningRate = startingLeagningRate

        for epoch in range(epochs):
            for featuresBatch, observedLabelsBatch in dataloader:

                featuresBatch = featuresBatch.to(self.device)
                observedLabelsBatch = observedLabelsBatch.to(self.device)

                predictedLabelsBatch = self.forward(featuresBatch)

                loss = torch.nn.functional.cross_entropy(predictedLabelsBatch, observedLabelsBatch)

                self.backward(featuresBatch, observedLabelsBatch, currentLearningRate)

            currentLearningRate = currentLearningRate - currentLearningRate * learningRateDecayPercentage

            epochEndTime = time.time()
            epochElapsedTime = epochEndTime - startTime
            totalLoss += loss.item()
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}, Time elapsed: {epochElapsedTime} seconds")

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
    def __init__(self, inputs: int, neurons: int, activation: Callable[[Tensor],Tensor], device: torch.device):

        # self.__validator = Validator(Layer)
        # self.__validator.validateActivation(activation, neurons)

        self.inputs = inputs
        self.neurons = neurons
        self.activation = activation

        self.weights = (1 + 1) * torch.rand(self.inputs, self.neurons, device=device) - 1
        self.biases = torch.rand(self.neurons, device=device)


    def forward(self, featureBatch: Tensor) -> Tensor:

        # self.__validator.validateTensorShapeAndSize(featureBatch, "featureBatch", self.inputs)

        z = torch.matmul(featureBatch, self.weights) + self.biases

        labels = self.activation(z)
        return labels
    
    def backward(self, inputLabelsBatch: Tensor, error: Tensor, learningRate: float):

        # self.__validator.validateTensorShapeAndSize(inputLabelsBatch, "inputLabelsBatch", self.inputs)
        # self.__validator.validateTensorShapeAndSize(error, "error", self.neurons)

        gradient = inputLabelsBatch.t() @ error #torch.matmul(inputLabelsBatch.t(), error)

        # self.weights -= learningRate * gradient
        self.weights.sub_(learningRate * gradient)

        # self.biases -= learningRate * error.sum(dim=0)
        self.biases.sub_(learningRate * error.mean(dim=0))
