import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, random_split
from torch import device
from model import ThreeLayerModel

class OneHotMNIST(Dataset):
    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset

    def __getitem__(self, index):
        image, label = self.mnist_dataset[index]
        labelOneHot = torch.zeros(10)
        labelOneHot[label] = 1
        imageFlat = image.view(-1)
        return imageFlat, labelOneHot

    def __len__(self):
        return len(self.mnist_dataset)

def getMNISTDataset() -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    
    dataset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
    dataset = OneHotMNIST(dataset)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

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

    model.train(trainingData, epochs=40, learningRate=0.002)

    accuracyAfterTraining = model.getAccuracy(testData)
    print(f"Accuracy after training is: {accuracyAfterTraining}")

if __name__ == "__main__":
    main()