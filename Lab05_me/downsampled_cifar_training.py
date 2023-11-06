import os
import random
from multiprocessing import freeze_support

import torch
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

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
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        # return self.fc2(self.relu(self.fc1(x)))
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def accuracy(output, labels):
    fp_plus_fn = torch.logical_not(output == labels).sum().item()
    all_elements = len(output)
    return (all_elements - fp_plus_fn) / all_elements


def train(model, train_loader, criterion, optimizer, device, writer, epoch_count):
    model.train()

    all_outputs = []
    all_labels = []

    batch_count = 0
    total_loss = 0
    for data, labels in train_loader:
        data = data.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        output = model(data)
        loss = criterion(output, labels)

        loss.backward()
        total_loss += loss
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        output = output.softmax(dim=1).detach().cpu().squeeze()
        labels = labels.cpu().squeeze()
        all_outputs.append(output)
        all_labels.append(labels)

        writer.add_scalar("Train/Loss/Batch", loss, epoch_count * len(train_loader) + batch_count)
        batch_count +=1

    writer.add_scalar("Train/Loss/Epoch", total_loss, epoch_count)

    all_outputs = torch.cat(all_outputs).argmax(dim=1)
    all_labels = torch.cat(all_labels)

    return round(accuracy(all_outputs, all_labels), 4)


def validate(model, val_loader, criterion, device ,writer, epoch_count):
    model.eval()

    all_outputs = []
    all_labels = []

    total_loss = 0
    for data, labels in val_loader:
        data = data.to(device, non_blocking=True)

        with torch.no_grad():
            output = model(data)

        output = output.to('cpu')

        loss = criterion(output, labels)
        total_loss += loss

        output = output.softmax(dim=1).squeeze()
        labels = labels.squeeze()
        all_outputs.append(output)
        all_labels.append(labels)

    writer.add_scalar("Val/Loss/Epoch", total_loss, epoch_count)

    all_outputs = torch.cat(all_outputs).argmax(dim=1)
    all_labels = torch.cat(all_labels)

    return round(accuracy(all_outputs, all_labels), 4)


def do_epoch(model, train_loader, val_loader, criterion, optimizer, scheduler, device, writer, epoch_count):
    acc = train(model, train_loader, criterion, optimizer,  device, writer, epoch_count)
    acc_val = validate(model, val_loader, criterion, device, writer, epoch_count)

    scheduler.step()

    # torch.cuda.empty_cache()
    return acc, acc_val


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

    model = MLP(784, 100, 10)
    model = model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.9)  # Adjust step_size and gamma as needed
    criterion = torch.nn.CrossEntropyLoss()
    epochs = 50

    batch_size = 256
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
        acc, acc_val = do_epoch(model, train_loader, val_loader, criterion, optimizer, scheduler, device, writer, epoch)

        tbar.set_postfix_str(f"Acc: {acc}, Acc_val: {acc_val}")

        #epoch training accuracy
        writer.add_scalar("Train/Accuracy", acc, epoch)

        writer.add_scalar("Train/Learning Rate", optimizer.param_groups[0]['lr'], epoch)

        #epoch validation accuracy
        writer.add_scalar("Val/Accuracy", acc_val, epoch)

        #epoch model norm
        writer.add_scalar("Model/Norm", get_model_norm(model), epoch)



if __name__ == '__main__':
    freeze_support()
    main()
