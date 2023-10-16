from torch import Tensor
import torch
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, Subset, random_split
from perceptron_layer import PerceptronLayer


def train_perceptron(X: Tensor, W: Tensor, b: Tensor, y_true: Tensor, mu: float):
    # Create a layer of perceptrons
    layer = PerceptronLayer(weights = W, biases = b, learningRate = mu)

    # Forward and backward propagation
    for x, y in zip(X, y_true):
        predictedY = layer.forward(x)
        layer.backward(x, y, predictedY)

    # Get the updated weights and biases
    W_updated = torch.stack([p.weights for p in layer.perceptrons]).t()
    b_updated = torch.tensor([p.bias for p in layer.perceptrons])

    return W_updated, b_updated


def getMNISTDataset() -> TensorDataset:
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

    trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    labels_one_hot = torch.zeros(len(labels), 10)
    labels_one_hot.scatter_(1, labels.unsqueeze(1), 1)

    images = images.view(images.shape[0], -1)

    return TensorDataset(images, labels_one_hot)

def splitDataSet(dataset: TensorDataset) -> tuple[Subset, Subset]:
    trainingDataLength = int(0.8 * len(dataset))
    testDataLength = len(dataset) - trainingDataLength
    trainingData, testData = random_split(dataset, [trainingDataLength, testDataLength])

    return trainingData, testData

def main():
    W = torch.randn(10, 784)
    b = torch.randn(10)

    nn = PerceptronLayer(weights=W, biases=b, learningRate=0.1)
    
    MNISTDataset = getMNISTDataset()

    trainingData, testData = splitDataSet(MNISTDataset)

    accuracyBeforeTraining = nn.getAccuracy(MNISTDataset)
    print(f"Accuracy before training is: {accuracyBeforeTraining}")

    epochs = 1000
    nn.train(trainingData, epochs)

    accuracyAfterTraining = nn.getAccuracy(MNISTDataset)
    print(f"Accuracy after training is: {accuracyAfterTraining}")

    accuracyOnTrainingData = nn.getAccuracy(trainingData)
    print(f"Accuracy for training data: {accuracyOnTrainingData}")

    accuracyOnNewData = nn.getAccuracy(testData)
    print(f"Accuracy for test (new) data: {accuracyOnNewData}")

    # x,y = MNISTDataset.tensors
    # print(train_perceptron(x,W,b,y,0.1))


if __name__ == "__main__":
    main()