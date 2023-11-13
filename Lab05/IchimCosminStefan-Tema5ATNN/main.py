import torch
import torchvision.transforms as transforms
import wandb
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2
from tqdm import tqdm
from Model import MLP
import numpy as np


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mos')
    return torch.device('cpu')


def accuracy(output, labels):
    fp_plus_fn = torch.logical_not(output == labels).sum().item()
    all_elements = len(output)
    return (all_elements - fp_plus_fn) / all_elements


def train(model, train_loader, criterion, optimizer, device):
    model.train()
    all_outputs = []
    all_labels = []
    running_loss = 0.0
    batch_training_losses = []
    for data, labels in train_loader:
        data = data.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        output = model(data)
        loss = criterion(output, labels)
        loss = loss.to(device, non_blocking=True)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        output = output.softmax(dim=1).detach().cpu().squeeze()
        labels = labels.cpu().squeeze()
        all_outputs.append(output)
        all_labels.append(labels)
        running_loss += loss.item()
        batch_training_losses.append(loss.item())
    all_outputs = torch.cat(all_outputs).argmax(dim=1)
    all_labels = torch.cat(all_labels)
    epoch_loss = running_loss / len(train_loader)
    return round(accuracy(all_outputs, all_labels), 4), epoch_loss, batch_training_losses


def validation(model, validation_loader, validation_criterion, device):
    model.eval()
    all_outputs = []
    all_labels = []
    running_loss = 0.0
    for data, labels in validation_loader:
        data = data.to(device, non_blocking=True)
        with torch.no_grad():
            output = model(data)
            output = output.softmax(dim=1).cpu().squeeze()
            loss = criterion(output, labels)
            running_loss += loss.item()
            labels = labels.squeeze()
            all_outputs.append(output)
            all_labels.append(labels)
    all_outputs = torch.cat(all_outputs).argmax(dim=1)
    all_labels = torch.cat(all_labels)
    validation_loss = running_loss / len(validation_loader)
    return round(accuracy(all_outputs, all_labels), 4), validation_loss


def one_epoch(model, train_loader, val_loader, criterion, optimizer, device):
    acc, epoch_loss, batch_training_losses = train(model, train_loader, criterion, optimizer, device)
    acc_val, validation_loss = validation(model, val_loader, criterion, device)
    torch.cuda.empty_cache()
    return acc, acc_val, epoch_loss, batch_training_losses, validation_loss


def get_model_norm(model):
    norm = 0.0
    for param in model.parameters():
        norm += torch.norm(param)
    return norm


if __name__ == '__main__':
    transforms = [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((28, 28), antialias=True),
        v2.Grayscale(),
        torch.flatten,
    ]
    data_path = '../data'
    train_dataset = CIFAR10(root=data_path, train=True, transform=v2.Compose(transforms), download=True)
    validation_dataset = CIFAR10(root=data_path, train=False, transform=v2.Compose(transforms), download=True)

    wandb.init(
        project="Tema5ATNN",
        config={
            "architecture": "MLP",
            "dataset": "CIFAR-10",
            "epochs": 50,
        }
    )
    batch_size = 128
    for learning_rate in np.arange(0.001, 0.11, 0.001):
        model = MLP(784, 100, 10)
        for optimizer in (torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9), torch.optim.Adam(model.parameters(), lr=learning_rate), torch.optim.RMSprop(model.parameters(), lr=learning_rate)):
            model = model.to(get_default_device())
            criterion = torch.nn.CrossEntropyLoss()
            epochs = 15
            val_batch_size = 50
            num_workers = 2
            persistent_workers = (num_workers != 0)
            pin_memory = get_default_device().type == 'cuda'
            train_loader = DataLoader(train_dataset, shuffle=True, pin_memory=pin_memory, num_workers=num_workers,
                                      batch_size=batch_size, drop_last=True, persistent_workers=persistent_workers)
            val_loader = DataLoader(validation_dataset, shuffle=False, pin_memory=True, num_workers=0, batch_size=val_batch_size,
                                    drop_last=False)
            writer = SummaryWriter()
            tbar = tqdm(tuple(range(epochs)))
            for epoch in tbar:
                running_loss = 0.0
                acc, acc_val, epoch_loss, batch_training_loses, validation_loss = one_epoch(model, train_loader, val_loader, criterion, optimizer, get_default_device())
                tbar.set_postfix_str(f"Acc: {acc}, Acc_val: {acc_val}")
                writer.add_scalar("Train/Accuracy", acc, epoch)
                writer.add_scalar("Val/Accuracy", acc_val, epoch)
                writer.add_scalar("Model/Norm", get_model_norm(model), epoch)
                writer.add_scalar("Train/Loss", epoch_loss, epoch)
                writer.add_scalar("Val/Loss", validation_loss, epoch)
                writer.add_scalar("Batch/Size", batch_size, epoch)
                writer.add_scalar("Learning/Rate", learning_rate, epoch)
                wandb.log({"Optimizer": type(optimizer).__name__, "Epoch": epoch, "Train/Accuracy": acc, "Val/Accuracy": acc_val, "Model/Norm": get_model_norm(model), "Train/Loss": epoch_loss, "Val/Loss": validation_loss, "Batch/Size": batch_size, "Learning/Rate": learning_rate})
                for pozitie, i in enumerate(batch_training_loses):
                    writer.add_scalar("Batch/Loss", i, epoch * batch_size + pozitie)
    wandb.finish()