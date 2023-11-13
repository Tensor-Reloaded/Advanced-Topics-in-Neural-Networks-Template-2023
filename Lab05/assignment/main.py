import os
from multiprocessing import freeze_support

import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import wandb

wandb.login(key='d6125cbd90b59143e378e1e5b0dfd7a8379e0d7f')

config = {'method': 'random',
          'metric': {'goal': 'maximize', 'name': 'acc_val'},
          'parameters': {'val_batch_size': {'distribution': 'q_log_uniform_values',
                                            'max': 256,
                                            'min': 32,
                                            'q': 8},
                         'epochs': {'value': 50},
                         'learning_rate': {'distribution': 'uniform',
                                           'max': 0.01,
                                           'min': 0},
                         # 'optimizer': {'value': 'adam'}
                         }
          }

sweep_id = wandb.sweep(config, project="carn-lab5")


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
        return self.fc2(self.relu(self.fc1(x)))
        # x = self.fc1(x)
        # x = self.relu(x)
        # x = self.fc2(x)
        # return x


def accuracy(output, labels):
    fp_plus_fn = torch.logical_not(output == labels).sum().item()
    all_elements = len(output)
    return (all_elements - fp_plus_fn) / all_elements


def train(model, train_loader, criterion, optimizer, device):
    model.train()

    all_outputs = []
    all_labels = []
    train_loss = 0
    batch_train_loss = []
    for data, labels in train_loader:
        data = data.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        output = model(data)
        loss = criterion(output, labels)

        loss.backward()

        # torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        train_loss += loss.item()
        batch_train_loss.append(loss.item())

        output = output.softmax(dim=1).detach().cpu().squeeze()
        labels = labels.cpu().squeeze()
        all_outputs.append(output)
        all_labels.append(labels)

    all_outputs = torch.cat(all_outputs).argmax(dim=1)
    all_labels = torch.cat(all_labels)

    return round(accuracy(all_outputs, all_labels), 4), train_loss, batch_train_loss


def val(model, val_loader, criterion, device):
    model.eval()

    all_outputs = []
    all_labels = []

    validation_loss = 0
    for data, labels in val_loader:
        data = data.to(device, non_blocking=True)

        with torch.no_grad():
            output = model(data)

        output = output.softmax(dim=1).cpu().squeeze()
        labels = labels.squeeze()

        loss = criterion(output, labels)
        validation_loss += loss.item()

        all_outputs.append(output)
        all_labels.append(labels)

    all_outputs = torch.cat(all_outputs).argmax(dim=1)
    all_labels = torch.cat(all_labels)

    return round(accuracy(all_outputs, all_labels), 4), validation_loss


def do_epoch(model, train_loader, val_loader, criterion, optimizer, device):
    acc, train_loss, batch_train_loss = train(model, train_loader, criterion, optimizer, device)
    acc_val, validation_loss = val(model, val_loader, criterion, device)
    # torch.cuda.empty_cache()
    return acc, acc_val, train_loss, validation_loss, batch_train_loss


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

    with wandb.init(config=None):
        config = wandb.config
        # optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
        # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        # optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate)
        optimizer = torch.optim.Adagrad(model.parameters(), lr=config.learning_rate)

        criterion = torch.nn.CrossEntropyLoss()
        epochs = config.epochs

        batch_size = 256
        val_batch_size = config.val_batch_size
        num_workers = 2
        persistent_workers = (num_workers != 0)
        pin_memory = device.type == 'cuda'
        train_loader = DataLoader(train_dataset, shuffle=True, pin_memory=pin_memory, num_workers=num_workers,
                                  batch_size=batch_size, drop_last=True, persistent_workers=persistent_workers)
        val_loader = DataLoader(val_dataset, shuffle=False, pin_memory=True, num_workers=0, batch_size=val_batch_size,
                                drop_last=False)

        writer = SummaryWriter()
        tbar = tqdm(tuple(range(epochs)))

        # writer.add_scalar("Constants/Optimizer", optimizer, 0)

        for epoch in tbar:
            acc, acc_val, train_loss, validation_loss, batch_train_loss = do_epoch(model, train_loader, val_loader,
                                                                                   criterion, optimizer, device)
            tbar.set_postfix_str(f"Acc: {acc}, Acc_val: {acc_val}")

            writer.add_scalar("Train/Loss", train_loss / len(train_loader), epoch)
            writer.add_scalar("Train/Accuracy", acc, epoch)

            writer.add_scalar("Val/Loss", validation_loss / len(val_loader), epoch)
            writer.add_scalar("Val/Accuracy", acc_val, epoch)

            writer.add_scalar("Model/Norm", get_model_norm(model), epoch)
            writer.add_scalar("Constants/Learning rate", config.learning_rate, epoch)
            writer.add_scalar("Constants/Batch size", val_batch_size, epoch)

            for b, l in enumerate(batch_train_loss):
                writer.add_scalar("Batch Train/Loss", l, b)

            wandb.log({"acc_val": acc_val, "epoch": epoch})

    writer.close()
    wandb.finish()


if __name__ == '__main__':
    freeze_support()
    wandb.agent(sweep_id, main, count=3)
