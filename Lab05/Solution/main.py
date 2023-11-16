import os
from multiprocessing import freeze_support
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from dataset import Cifar10Dataset
from model import ImageModel
import matplotlib.pyplot as plt

import wandb

wandb.login(key="044dcd6db8b59791c6b07c29f846735bc94bd5ae")
wandb.init(
    project="lab5",
    config={
        "seed": 300,
        "lr": 0.09,
        "dataset": "cifar-10",
        "epochNumber": 50,
    }
)

def plot_validation_accuracies(accuracies_dict, plot_name):
    plt.figure(figsize=(10, 6))

    for optimizer, (epochs, accuracies) in accuracies_dict.items():
        plt.plot(epochs, accuracies, label=optimizer)

    plt.title('Validation Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the plot with the specified name
    plt.savefig(plot_name)

    # Show the plot (optional)
    plt.show()

def calculate_accuracy(outputs, targets):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == targets).sum().item()
    accuracy = correct / targets.size(0)
    return accuracy

def train(model, optimizer, criterion, train_loader, device, writer, epoch):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        targets_onehot = F.one_hot(targets, num_classes=10)

        loss = criterion(outputs, targets_onehot.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == targets).sum().item()
        total_samples += targets.size(0)

    accuracy = correct_predictions / total_samples
    batch_loss = total_loss / len(train_loader)

    return batch_loss, accuracy

def val(model, criterion, val_loader, device, writer, epoch):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == targets).sum().item()
            total_samples += targets.size(0)

    accuracy = correct_predictions / total_samples
    epoch_loss = total_loss / len(val_loader)

    writer.add_scalar('Epoch Validation Loss', epoch_loss, epoch)
    writer.add_scalar('Epoch Validation Accuracy', accuracy, epoch)

    return epoch_loss, accuracy

def run(model, optimizer, optimizer_name, criterion, train_loader, val_loader, device, num_epochs, batch_size, val_batch_size, writer):
    accuracies_dict = {optimizer_name: ([], [])}
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(num_epochs):
        train_loss, train_accuracy = train(model, optimizer, criterion, train_loader, device, writer, epoch)
        val_loss, val_accuracy = val(model, criterion, val_loader, device, writer, epoch)

        accuracies_dict[str(optimizer_name)][0].append(epoch)
        accuracies_dict[str(optimizer_name)][1].append(val_accuracy)

        writer.add_scalar('Epoch number', epoch + 1, epoch)
        writer.add_scalar('Epoch Training Loss', train_loss, epoch)
        writer.add_scalar('Epoch Training Accuracy', train_accuracy, epoch)
        writer.add_scalar('Epoch Validation Loss', val_loss, epoch)
        writer.add_scalar('Epoch Validation Accuracy', val_accuracy, epoch)
        writer.add_scalar("Learning Rate", optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar("Batch Size", batch_size, epoch)
        writer.add_scalar("Validation Batch Size", val_batch_size, epoch)

        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

    plot_validation_accuracies(accuracies_dict, optimizer_name)

    return train_losses, val_losses, train_accuracies, val_accuracies

def get_optimizer(model, optimizer_name, learning_rate):
    if optimizer_name == "SGD":
        return torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_name == "Adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == "RMSProp":
        return torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    elif optimizer_name == "AdaGrad":
        return torch.optim.Adagrad(model.parameters(), lr=learning_rate)
    elif optimizer_name == "SGD_SAM":
        base_optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        return SAM(base_optimizer, rho=0.05, adaptive=True)

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    transforms = [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((28, 28), antialias=True),
        v2.Grayscale(),
        torch.flatten,
    ]

    criterion = nn.CrossEntropyLoss()

    data_path = '../data'
    train_dataset = CIFAR10(root=data_path, train=True, transform=v2.Compose(transforms), download=True)
    val_dataset = CIFAR10(root=data_path, train=False, transform=v2.Compose(transforms), download=True)
    train_dataset = Cifar10Dataset(train_dataset)
    val_dataset = Cifar10Dataset(val_dataset)
    model = ImageModel(784, 100, 10)
    model = model.to(device)

    config = wandb.config
    epochs = config.epochNumber
    batch_size = 128
    val_batch_size = 500

    for optimizer_name in ["SGD", "Adam", "RMSProp", "AdaGrad", "SGD_SAM"]:
        optimizer = get_optimizer(model, optimizer_name, config.lr)

        writer = SummaryWriter(comment=f'_{optimizer_name}')

        pin_memory = device.type == 'cuda'
        train_loader = DataLoader(train_dataset, shuffle=True, pin_memory=pin_memory,
                                  batch_size=batch_size, drop_last=True)
        val_loader = DataLoader(val_dataset, shuffle=False, pin_memory=True, num_workers=0, batch_size=val_batch_size,
                                drop_last=False)

        train_losses, val_losses, train_accuracies, val_accuracies = run(
            model, optimizer, optimizer_name, criterion, train_loader, val_loader, device, epochs, batch_size, val_batch_size, writer
        )

        wandb.finish()