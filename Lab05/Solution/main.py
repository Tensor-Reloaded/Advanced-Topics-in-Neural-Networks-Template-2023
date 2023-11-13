from multiprocessing import freeze_support

import torch
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
    def __init__(self, dataset, transforms=None, cache=True):
        if cache:
            dataset = tuple([x for x in dataset])
        self.dataset = dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        x = self.dataset[i][0]
        if self.transforms:
            x = self.transforms(x)
        y = torch.zeros(10)
        y[self.dataset[i][1]] = 1
        return x, y


class MLP(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_size, 2 * input_size),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(2 * input_size, 4 * input_size),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(4 * input_size, 8 * input_size),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(8 * input_size, 4 * input_size),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(4 * input_size, 2 * input_size),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(2 * input_size, input_size),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(input_size, output_size),

        )

    def forward(self, x):
        # return self.fc2(self.relu(self.fc1(x)))
        x = self.network(x)

        return x


def accuracy(output, labels):
    fp_plus_fn = torch.logical_not(output == labels).sum().item()
    all_elements = len(output)
    return (all_elements - fp_plus_fn) / all_elements


def train(model, train_loader, criterion, optimizer, scheduler, writer, epoch, device):
    model.train()

    all_outputs = []
    all_labels = []
    total_loss = 0
    ite = 0
    for data, labels in train_loader:
        optimizer.zero_grad()
        data = data.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        output = model(data)
        # print(labels.shape)
        # print(output.shape)
        loss = criterion(output, labels)
        # print(loss)
        ite += 1
        loss.backward()
        total_loss += loss.item()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        if writer:
            writer.add_scalar("Train/Batch_Loss", loss.item(), epoch * len(train_loader) + ite)
        optimizer.step()
        scheduler.step()

        output = output.softmax(dim=1).detach().cpu().squeeze()
        # print(output.shape)
        labels = labels.cpu().squeeze()
        all_outputs.append(output)
        all_labels.append(labels)
    total_loss /= len(train_loader)
    all_outputs = torch.cat(all_outputs).argmax(dim=1)
    all_labels = torch.cat(all_labels).argmax(dim=1)

    return round(accuracy(all_outputs, all_labels), 4), total_loss


def val(model, val_loader, criterion, device):
    model.eval()

    all_outputs = []
    all_labels = []
    total_loss = 0
    for data, labels in val_loader:
        data = data.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.no_grad():
            output = model(data)
        loss = criterion(output, labels)
        total_loss += loss.item()
        output = output.softmax(dim=1).cpu().squeeze()
        labels = labels.cpu().squeeze()  # .to(torch.bool)
        all_outputs.append(output)
        all_labels.append(labels)
    total_loss /= len(val_loader)
    all_outputs = torch.cat(all_outputs).argmax(dim=1)
    all_labels = torch.cat(all_labels).argmax(dim=1)  # .to(torch.bool)
    print(all_labels)

    return round(accuracy(all_outputs, all_labels), 4), total_loss


def do_epoch(model, train_loader, val_loader, criterion, optimizer, scheduler, writer, epoch, device):
    acc, train_loss = train(model, train_loader, criterion, optimizer, scheduler, writer, epoch, device)
    acc_val, val_loss = val(model, val_loader, criterion, device)
    # torch.cuda.empty_cache()
    return acc, acc_val, train_loss, val_loss


def get_model_norm(model):
    norm = 0.0
    for param in model.parameters():
        norm += torch.norm(param)
    return norm


def main(device=get_default_device()):
    main_transforms = [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.49139968, 0.48215827, 0.44653124], std=[0.24703233, 0.24348505, 0.26158768]),
        v2.Resize((28, 28), antialias=True),
        v2.Grayscale(),

    ]
    train_transforms = [
        v2.RandomAffine(degrees=10, translate=(0.0, 0.1), scale=(0.8, 1.0)),
        v2.RandomHorizontalFlip(0.25),
        v2.RandomVerticalFlip(0.25),
        torch.flatten,
    ]
    valid_transforms = [
        torch.flatten,
    ]
    # train_transforms = [
    # v2.RandomAffine(degrees=10, translate=(0.0, 0.1), scale=(0.8, 1.0)),
    # v2.RandomHorizontalFlip(0.25),
    # v2.RandomVerticalFlip(0.25),
    # v2.Normalize(mean=[0.49139968, 0.48215827 ,0.44653124], std=[0.24703233, 0.24348505, 0.26158768]),
    # v2.Resize((28, 28), antialias=True),
    # v2.Grayscale(),
    # torch.flatten,
    # ]
    # valid_transforms = [
    #     v2.Normalize(mean=[0.49139968, 0.48215827 ,0.44653124], std=[0.24703233, 0.24348505, 0.26158768]),
    #     v2.Resize((28, 28), antialias=True),
    #     v2.Grayscale(),
    #     torch.flatten,
    # ]

    data_path = '../data'
    train_dataset = CIFAR10(root=data_path, train=True, transform=v2.Compose(main_transforms), download=True)
    val_dataset = CIFAR10(root=data_path, train=False, transform=v2.Compose(main_transforms), download=True)
    train_dataset = CachedDataset(train_dataset, transforms=v2.Compose(train_transforms))
    val_dataset = CachedDataset(val_dataset, transforms=v2.Compose(valid_transforms))

    model = MLP(784, 10)
    model = model.to(device)
    learning_rate = 1e-3
    dropout = 0.3
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    epochs = 2000
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)

    batch_size = 256
    val_batch_size = 500
    num_workers = 2
    persistent_workers = (num_workers != 0)
    pin_memory = device.type == 'cuda'
    train_loader = DataLoader(train_dataset, shuffle=True, pin_memory=pin_memory, num_workers=num_workers,
                              batch_size=batch_size, drop_last=True, persistent_workers=persistent_workers)
    val_loader = DataLoader(val_dataset, shuffle=False, pin_memory=True, num_workers=0, batch_size=val_batch_size,
                            drop_last=False)
    max_val = 0
    writer = SummaryWriter()
    tbar = tqdm(tuple(range(epochs)))
    for epoch in tbar:
        acc, acc_val, train_loss, val_loss = do_epoch(model, train_loader, val_loader, criterion, optimizer, scheduler,
                                                      writer, epoch, device)
        if acc_val > max_val:
            max_val = acc_val
        tbar.set_postfix_str(f"Acc: {acc}, Acc_val: {acc_val}")
        writer.add_scalar("Train/Accuracy", acc, epoch)
        writer.add_scalar("Val/Accuracy", acc_val, epoch)
        writer.add_scalar("Train/Loss", train_loss, epoch)
        writer.add_scalar("Val/Loss", val_loss, epoch)
        writer.add_scalar("Model/Norm", get_model_norm(model), epoch)
    writer.add_hparams(
        {'lr': learning_rate, 'batch_size': batch_size, 'optimizer': optimizer.__class__.__name__, 'epochs': epochs,
         'dropout': dropout}, {'val_acc': max_val})


if __name__ == '__main__':
    freeze_support()
    main()