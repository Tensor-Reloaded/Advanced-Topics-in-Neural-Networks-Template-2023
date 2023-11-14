import os
import random
from multiprocessing import freeze_support

import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import wandb

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
    def __init__(self, input_size, hidden_size, output_size, dropout_prob):
        super(MLP, self).__init__()
       # self.fc1 = torch.nn.Linear(input_size, hidden_size)
       # self.fc2 = torch.nn.Linear(hidden_size, output_size)
       # self.relu = torch.nn.ReLU(inplace=True)

        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.bn1 = torch.nn.BatchNorm1d(hidden_size)
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.dropout1 = torch.nn.Dropout(p=dropout_prob)

        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.bn2 = torch.nn.BatchNorm1d(hidden_size)
        self.relu2 = torch.nn.ReLU(inplace=True)
        self.dropout2 = torch.nn.Dropout(p=dropout_prob)

        self.fc3 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # return self.fc2(self.relu(self.fc1(x)))
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
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

def choose_optimizer(model, optimizer_choice, learning_rate):
    if optimizer_choice == 'sgd':
        return optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_choice == 'adam':
        return optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_choice == 'rmsprop':
        return optim.RMSprop(model.parameters(), lr=learning_rate)
    elif optimizer_choice == 'adagrad':
        return optim.Adagrad(model.parameters(), lr=learning_rate)
  #  elif optimizer_choice == 'sgd_sam':
        # Assuming you have the SAM (Sharpness-Aware Minimization) optimizer implementation
        # You can find it here: https://github.com/davda54/sam
   #     from sam import SAM
    #    base_optimizer = optim.SGD
    #    return SAM(model.parameters(), base_optimizer, lr=learning_rate)



def main(device=get_default_device()):
    #Hyperparameters tuning

    wandb.init(
        project='lab5',
        config={
            'scheduler_gamma': 0.90,
            'scheduler_step_size': 2,
            'batch_size': 64,
            'dropout_prob': 0.01,
            'optimizer_choice': 'adam',
            'learning_rate': 0.5
        }
    )

    config = wandb.config
    scheduler_gamma = config.scheduler_gamma
    scheduler_step_size = config.scheduler_step_size
    batch_size = config.batch_size
    dropout_prob = config.dropout_prob
    optimizer_choice = config.optimizer_choice
    learning_rate = config.learning_rate


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

    model = MLP(784, 100, 10, dropout_prob)
    model = model.to(device)

    optimizer = choose_optimizer(model, optimizer_choice,  learning_rate)
   # optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(beta1, beta2))
    scheduler = StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)  # Adjust step_size and gamma as needed
    criterion = torch.nn.CrossEntropyLoss()
    epochs = 150

    #batch_size = 256
    val_batch_size = 500
    num_workers = 10

    persistent_workers = (num_workers != 0)
    pin_memory = device.type == 'cuda'
    train_loader = DataLoader(train_dataset, shuffle=True, pin_memory=pin_memory, num_workers=num_workers,
                              batch_size=batch_size, drop_last=True, persistent_workers=persistent_workers)
    val_loader = DataLoader(val_dataset, shuffle=False, pin_memory=True, num_workers=0, batch_size=val_batch_size,
                            drop_last=False)

    writer = SummaryWriter()
    tbar = tqdm(tuple(range(epochs)))

    #global logging
    optimizer_info = f'Optimizer: {optimizer_choice}, Learning Rate: {learning_rate}'
    writer.add_text('Train/Optimizer Info',  optimizer_info)
    writer.add_text('Train/Batch Size',   f'Batch size: { batch_size} ')

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


    wandb.log({'accuracy': acc_val})
    wandb.log({'scheduler_gamma': scheduler_gamma})
    wandb.log({'scheduler_step_size': scheduler_step_size})
    wandb.log({'batch_size': batch_size})
    wandb.log({'optimizer_choice': optimizer_choice})


if __name__ == '__main__':
    freeze_support()
    main()
