import os
from multiprocessing import freeze_support

import torch
from torch.optim.lr_scheduler import StepLR
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import wandb

wandb.init(
    # set the wandb project where this run will be logged
    project="cifar10-analysis-project",

    # track hyperparameters and run metadata
    config={
        "learning_rate": 0.001,
        "architecture": "MLP",
        "dataset": "CIFAR-10",
        "epochs": 100,
    }
)


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
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, output_size)
        self.relu = torch.nn.ReLU(inplace=True)

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


def train(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    epoch_loss = 0.0
    batch_losses = []

    all_outputs = []
    all_labels = []

    for batch_idx, (data, labels) in enumerate(train_loader):
        data = data.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        output = model(data)
        loss = criterion(output, labels)

        loss.backward()

        # torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

        sharpness = calculate_gradient_sharpness(model)

        for param in model.parameters():
            param.grad = param.grad / (sharpness + 1e-10)  # Add a small constant to prevent division by zero

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        output = output.softmax(dim=1).detach().cpu().squeeze()
        labels = labels.cpu().squeeze()
        all_outputs.append(output)
        all_labels.append(labels)

        epoch_loss += loss.item()
        batch_losses.append((loss.item(), epoch * len(train_loader) + batch_idx))

        optimizer.step()

    all_outputs = torch.cat(all_outputs).argmax(dim=1)
    all_labels = torch.cat(all_labels)

    avg_epoch_loss = epoch_loss / len(train_loader)

    acc = accuracy(all_outputs, all_labels)
    wandb.log({'acc': acc, 'loss': avg_epoch_loss})

    return round(acc, 4), avg_epoch_loss, batch_losses


def val(model, val_loader, device, epoch):
    model.eval()

    all_outputs = []
    all_labels = []

    epoch_val_loss = 0.0
    batch_val_losses = []

    for batch_idx, (data, labels) in enumerate(val_loader):
        data = data.to(device, non_blocking=True)

        with torch.no_grad():
            output = model(data)

        output = output.softmax(dim=1).cpu().squeeze()
        labels = labels.squeeze()
        all_outputs.append(output)
        all_labels.append(labels)

        loss = torch.nn.CrossEntropyLoss()(output, labels)
        epoch_val_loss += loss.item()
        batch_val_losses.append((loss.item(), epoch * len(val_loader) + batch_idx))


    all_outputs = torch.cat(all_outputs).argmax(dim=1)
    all_labels = torch.cat(all_labels)

    avg_epoch_val_loss = epoch_val_loss / len(val_loader)

    acc = accuracy(all_outputs, all_labels)

    wandb.log({'val_acc': acc, 'val_loss': avg_epoch_val_loss})

    return round(acc, 4), avg_epoch_val_loss, batch_val_losses


def do_epoch(model, train_loader, val_loader, criterion, optimizer, device, epoch):
    acc, epoch_loss, batch_loss = train(model, train_loader, criterion, optimizer, device, epoch)
    acc_val, epoch_val_loss, batch_val_loss = val(model, val_loader, device, epoch)
    # torch.cuda.empty_cache()
    return acc, acc_val, epoch_loss, epoch_val_loss, batch_loss, batch_val_loss


def get_model_norm(model):
    norm = 0.0
    for param in model.parameters():
        norm += torch.norm(param)
    return norm


def calculate_gradient_sharpness(model):
    sharpness = 0.0
    for param in model.parameters():
        sharpness += torch.norm(param.grad)
    return sharpness


def main(device=get_default_device()):
    transforms = [
        v2.ToImageTensor(),
        v2.ToDtype(torch.float32),
        v2.Resize((28, 28), antialias=True),
        v2.Grayscale(),
        torch.flatten,
    ]

    data_path = '../data'
    train_dataset = CIFAR10(root=data_path, train=True, transform=v2.Compose(transforms), download=True)
    val_dataset = CIFAR10(root=data_path, train=False, transform=v2.Compose(transforms), download=True)
    train_dataset = CachedDataset(train_dataset)
    val_dataset = CachedDataset(val_dataset)

    model = MLP(784, 10)
    model = model.to(device)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
    # optimizer = torch.optim.Adagrad(model.parameters(), lr=0.001)

    momentum = 0.9
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=momentum)
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.5)


    criterion = torch.nn.CrossEntropyLoss()
    epochs = 100

    batch_size = 128
    val_batch_size = 500
    num_workers = 2
    persistent_workers = (num_workers != 0)
    pin_memory = device.type == 'cuda'
    train_loader = DataLoader(train_dataset, shuffle=True, pin_memory=pin_memory, num_workers=num_workers,
                              batch_size=batch_size, drop_last=True, persistent_workers=persistent_workers)
    val_loader = DataLoader(val_dataset, shuffle=False, pin_memory=True, num_workers=0, batch_size=val_batch_size,
                            drop_last=False)

    writer = SummaryWriter()
    tbar = tqdm(tuple(range(epochs)))
    for epoch in tbar:
        acc, acc_val, epoch_loss, epoch_val_loss, batch_loss, batch_val_loss = \
            do_epoch(model, train_loader, val_loader, criterion, optimizer, device, epoch)
        tbar.set_postfix_str(f"Acc: {acc}, Acc_val: {acc_val}")
        writer.add_scalar("Train/Accuracy", acc, epoch)
        writer.add_scalar("Val/Accuracy", acc_val, epoch)
        writer.add_scalar("Model/Norm", get_model_norm(model), epoch)

        average_batch_loss = sum([x[0] for x in batch_loss]) / len(batch_loss)
        average_batch_val_loss = sum([x[0] for x in batch_val_loss]) / len(batch_val_loss)

        writer.add_scalar("Train/BatchLoss", average_batch_loss, epoch)
        writer.add_scalar("Val/BatchLoss", average_batch_val_loss, epoch)

        writer.add_scalar("Train/EpochLoss", epoch_loss, epoch)
        writer.add_scalar("Val/EpochLoss", epoch_val_loss, epoch)

        writer.add_scalar("Hyperparameters/LearningRate", optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar("Hyperparameters/BatchSize", batch_size, epoch)
        writer.add_scalar("Hyperparameters/ValBatchSize", val_batch_size, epoch)

        optimizer_str = str(optimizer.__class__).split('.')[-1].split("'")[0] + " with momentum " + str(momentum) + " and SAM"
        writer.add_text("Hyperparameters/Optimizer", optimizer_str, epoch)


if __name__ == '__main__':
    freeze_support()
    main()

# Best Val Acc: 0.4145
# Config: {'learning_rate': 0.001, 'architecture': 'MLP', 'dataset': 'CIFAR-10', 'epochs': 100}
# Optimizer: SGD with momentum 0.9 and SAM
# Batch Size: 128
# Val Batch Size: 500

# MLP config
# input_size = 784
# hidden1_size = 512
# hidden2_size = 256
# output_size = 10
# activations: ReLU (hidden layers), no activation (output layer)
# loss: CrossEntropyLoss
