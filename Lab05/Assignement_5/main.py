import os
from multiprocessing import freeze_support
import torch
import torchvision
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import wandb
import random
import torch.optim as optim
from sam import SAM

def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
        # For multi-gpu workstations, PyTorch will use the first available GPU (cuda:0), unless specified otherwise
        # (cuda:1).
    if torch.backends.mps.is_available():
        return torch.device('mos')
    return torch.device('cpu')


class CachedDataset(Dataset):
    def __init__(self, dataset, cache=True):
        if cache:
            dataset = tuple([x for x in dataset])
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return self.dataset[i]


class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size_1)
        self.fc2 = torch.nn.Linear(hidden_size_1, hidden_size_2)
        self.fc3 = torch.nn.Linear(hidden_size_2, output_size)
        self.relu = torch.nn.ReLU(inplace=True)
        self.dropout = torch.nn.Dropout(0.25)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)

        return x


def accuracy(output, labels):
    fp_plus_fn = torch.logical_not(output == labels).sum().item()
    all_elements = len(output)
    return (all_elements - fp_plus_fn) / all_elements


def train(model, train_loader, criterion, optimizer, device):
    model.train()

    loss_per_batch = []
    all_outputs = []
    all_labels = []
    loss_value = 0

    for data, labels in train_loader:
        data = data.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        output = model(data)
        loss = criterion(output, labels)

        loss_value += loss.item()

        loss_per_batch.append(round(loss.item(), 4))

        loss.backward()

        optimizer.step()

        optimizer.zero_grad(set_to_none=True)

        output = output.softmax(dim=1).detach().cpu().squeeze()
        labels = labels.cpu().squeeze()
        all_outputs.append(output)
        all_labels.append(labels)

    all_outputs = torch.cat(all_outputs).argmax(dim=1)
    all_labels = torch.cat(all_labels)

    return (round(accuracy(all_outputs, all_labels), 4)), round((loss_value/len(train_loader)), 4), loss_per_batch


def val(model, val_loader, criterion, device):
    model.eval()

    all_outputs = []
    all_labels = []
    loss_value = 0

    for data, labels in val_loader:
        data = data.to(device, non_blocking=True)

        with torch.no_grad():
            output = model(data)

            loss = criterion(output, labels)
            loss_value += loss.item()

        output = output.softmax(dim=1).cpu().squeeze()
        labels = labels.squeeze()
        all_outputs.append(output)
        all_labels.append(labels)

    all_outputs = torch.cat(all_outputs).argmax(dim=1)
    all_labels = torch.cat(all_labels)

    return round(accuracy(all_outputs, all_labels), 4), round((loss_value/len(val_loader)), 4)


def do_epoch(model, train_loader, val_loader, criterion, optimizer, device):
    acc, train_loss, loss_per_batch = train(model, train_loader, criterion, optimizer, device)
    acc_val, val_loss = val(model, val_loader, criterion, device)
    torch.cuda.empty_cache()
    return acc, train_loss, loss_per_batch, acc_val, val_loss


def get_model_norm(model):
    norm = 0.0
    for param in model.parameters():
        norm += torch.norm(param)
    return norm


def main(device=get_default_device()):
    transforms = [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((28, 28), antialias=True),
        v2.Grayscale(),
        torch.flatten,
    ]

    data_path = '../data'
    train_dataset = CIFAR10(root=data_path, train=True, transform=v2.Compose(transforms), download=True)

    val_dataset = CIFAR10(root=data_path, train=False, transform=v2.Compose(transforms), download=True)
    train_dataset = CachedDataset(train_dataset)
    val_dataset = CachedDataset(val_dataset)


    learning_rate_1 = 0.01  # 1 -> 0.0005; 2 -> 0.01; 3 -> 0.01
    learning_rate_2 = 0.01  # 1 -> 0.0005; 2 -> 0.01; 3 -> 0.00001

    model = MLP(784, 100, 50, 10)
    model = model.to(device)


    # Adam and SGD
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_1)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate_2)


    # RMSProp and AdaGrad
    # optimizer = torch.optim.Rprop(model.parameters(), lr=learning_rate_1)
    # optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate_2)


    # SGD with SAM
    # optimizer = SAM(model.parameters(), base_optimizer=optim.SGD, lr=learning_rate_2, momentum=0.9)


    criterion = torch.nn.CrossEntropyLoss()
    epochs = 100
    batch_size = 256
    val_batch_size = 500
    num_workers = 2
    persistent_workers = (num_workers != 0)
    pin_memory = device.type == 'cuda'
    train_loader = DataLoader(train_dataset, shuffle=True, pin_memory=pin_memory, num_workers=num_workers,
                              batch_size=batch_size, drop_last=True, persistent_workers=persistent_workers)
    val_loader = DataLoader(val_dataset, shuffle=False, pin_memory=True, num_workers=0, batch_size=val_batch_size,
                            drop_last=False)

    # tensorboard
    writer = SummaryWriter('runs/Optimizer_Adam')
    writer.add_scalar("Learning rate Adam", learning_rate_1)
    writer.add_scalar("Batch size train", batch_size)
    writer.add_scalar("Batch size val", val_batch_size)

    # wandb
    # start a new wandb run to track this script
    wandb.init(
        project="Homework_5",

        # track hyperparameters and run metadata
        config={
            "dataset": "CIFAR-100",
            "epochs": 100,
            "Optimizer Adam - learning_rate": 0.0005,
            "Train Batch size": 256,
            "Validation Batch size": 500
        }
    )
    '''
        config={
            "dataset": "CIFAR-100",
            "epochs": 100,
            "Optimizer SGD - learning_rate": 0.0005,
            "Optimizer Adam - learning_rate": 0.0005,
            "Train Batch size": 256,
            "Validation Batch size": 500
        }
        
        config={
            "dataset": "CIFAR-100",
            "epochs": 100,
            "Optimizer RMSProp - learning_rate": 0.01,
            "Optimizer AdaGrad - learning_rate": 0.01,
            "Train Batch size": 256,
            "Validation Batch size": 500
        }
        
        config={
            "dataset": "CIFAR-100",
            "epochs": 100,
            "Optimizer SGD with SAM - learning_rate": 0.00001,
            "Train Batch size": 256,
            "Validation Batch size": 500
        }
    '''

    tbar = tqdm(tuple(range(epochs)))
    all_loss_per_batch = []
    for epoch in tbar:
        print('Epoch: ', epoch)
        acc, train_loss, loss_per_batch, acc_val, val_loss = do_epoch(model, train_loader, val_loader, criterion, optimizer, device)
        tbar.set_postfix_str(f"Acc: {acc}, Train loss: {train_loss}, Acc_val: {acc_val}, Validation loss: {val_loss}")

        # log metrics to wandb
        wandb.log({"train_acc": acc, "train_loss": train_loss})
        wandb.log({"validation_acc": acc_val, "validation_loss": val_loss})
        wandb.log({"Model/Norm": get_model_norm(model)})

        writer.add_scalar("Train/Accuracy", acc, epoch)
        writer.add_scalar("Validation/Accuracy", acc_val, epoch)
        writer.add_scalar("Train/Loss", train_loss, epoch)
        writer.add_scalar("Validation/Loss", val_loss, epoch)
        writer.add_scalar("Model/Norm", get_model_norm(model), epoch)

        for l in range(len(loss_per_batch)):
            all_loss_per_batch.append(loss_per_batch[l])

    for lo in range(len(all_loss_per_batch)):
        writer.add_scalar("Train/Loss_Per_Batch", all_loss_per_batch[lo], lo)
        wandb.log({"loss_per_batch": all_loss_per_batch[lo]})

    wandb.finish()


if __name__ == '__main__':
    freeze_support()
    main()
