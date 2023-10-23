import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, random_split
from torch import device
from model import ThreeLayerModel

class OneHotMNIST(Dataset):
    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset
        self.one_hot_labels = self.__one_hot_encode_labels(mnist_dataset)

    def __one_hot_encode_labels(self, mnist_dataset):
        labels = [label for _, label in mnist_dataset]
        one_hot_labels = torch.zeros(len(labels), 10)
        one_hot_labels[torch.arange(len(labels)), labels] = 1
        return one_hot_labels

    def __getitem__(self, index):
        image, label = self.mnist_dataset[index]
        labelOneHot = self.one_hot_labels[index]
        imageFlat = image.view(-1)
        return imageFlat, labelOneHot

    def __len__(self):
        return len(self.mnist_dataset)

def getMNISTDataset() -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    
    dataset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
    dataset = OneHotMNIST(dataset)

    trainSize = int(0.8 * len(dataset))
    testSize = len(dataset) - trainSize
    trainDataset, testDataset = random_split(dataset, [trainSize, testSize])

    batchSize = 64
    trainloader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True)
    testloader = DataLoader(testDataset, batch_size=batchSize, shuffle=True)

    return trainloader, testloader

def getDevice() -> device:
    if torch.cuda.is_available():
        return device('cuda')
    else:
        return device('cpu')

def main():  
    trainingData, testData = getMNISTDataset()

    model = ThreeLayerModel(784,100,10,getDevice())

    accuracyBeforeTraining = model.getAccuracy(testData)
    print(f"Accuracy before training is: {accuracyBeforeTraining}")

    model.train(trainingData, epochs=20, startingLeagningRate=0.003, learningRateDecayPercentage=0.01)

    accuracyAfterTraining = model.getAccuracy(testData)
    print(f"Accuracy after training is: {accuracyAfterTraining}")

if __name__ == "__main__":
    main()