import wandb
import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
from data import CustomDataset
from model import TransformationModel
from utils import ground_truth_transform
from train import train_model


def main():
    # Initialize Weights & Biases
    wandb.init(project="ImageTransformationBenchmark", settings=wandb.Settings(start_method="thread"))

    model = TransformationModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load CIFAR10 dataset
    full_dataset = CustomDataset(data_path='./data', train=True, transform=ground_truth_transform)

    # Split dataset into training and validation sets
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # DataLoaders for training and validation
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Loss function and optimizer
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, device, train_loader, val_loader, loss_function, optimizer, epochs=10)

    # Save model weights
    torch.save(model.state_dict(), 'model_weights.pth')
    wandb.finish()


if __name__ == '__main__':
    main()
