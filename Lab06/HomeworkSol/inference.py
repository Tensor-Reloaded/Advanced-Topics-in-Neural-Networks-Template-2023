import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from model_utils import initialize_resnet


def load_model(model_path, model_name, num_classes):
    # Initialize the model
    model = initialize_resnet(model_name, num_classes, use_pretrained=False, feature_extract=False)
    # Load the saved weights
    model.load_state_dict(torch.load(model_path))

    return model


def evaluate_model(model, testloader, device):
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_path = "/content/best_model_val_accuracy_cifar10.pth"
    model_name = 'ResNet34'  # Change as needed
    dataset_name = 'CIFAR10'  # Change to CIFAR100 if needed
    num_classes = 10 if dataset_name == 'CIFAR10' else 100

    # Define the transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Load the dataset
    testset = getattr(torchvision.datasets, dataset_name)(
        root='./data', train=False, download=True, transform=transform)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    # Load the model
    model = load_model(model_path, model_name, num_classes)

    # Evaluate the model
    test_accuracy = evaluate_model(model, testloader, device)
    print(f'Test Accuracy: {test_accuracy}%')

if __name__ == '__main__':
    main()
