import random
from multiprocessing import freeze_support

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2
from tqdm import tqdm

from model import ConvNet


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


class TrainCachedDataset(Dataset):
    def __init__(self, dataset, cache=True, device='cuda'):
        self.dataset = dataset
        self.augmentation = self._get_default_augmentation()
        if cache:
            self.cached_data = [x for x in dataset]

        self.device = device
        self.to_device()

    def _get_default_augmentation(self):
        return v2.Compose([
            v2.RandomApply([
                v2.RandomHorizontalFlip(p=1),
                v2.RandomVerticalFlip(p=1),
                v2.RandomRotation(degrees=(-30, 30), expand=False, center=None),
                v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                v2.RandomResizedCrop(size=(28, 28), scale=(0.8, 1.0), ratio=(0.8, 1.2), antialias=True),
            ], p=0.5),
        ])

    def _augment_data(self, data):
        image, label = data
        augmented_image = self._augmentation(image)
        return augmented_image, label

    def _augmentation(self, image):
        return self.augmentation(image)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        # if random.randint(0, 2) == 0:
        #     augmented_image, label = self._augment_data(self.dataset[i])

        #     return augmented_image, label
        return self.dataset[1]

    def set_device(self, device):
        self.device = device
        self.to_device()

    def to_device(self):
        if self.device == 'cuda':
            self.augmentation = self.augmentation.cuda()
        else:
            self.augmentation = self.augmentation.cpu()

    def set_augmentation(self, augmentation):
        self.augmentation = augmentation
        self.to_device()


def accuracy(output, labels):
    fp_plus_fn = torch.logical_not(output == labels).sum().item()
    all_elements = len(output)
    return (all_elements - fp_plus_fn) / all_elements


def train(model, train_loader, criterion, optimizer, device):
    model.train()

    all_outputs = []
    all_labels = []

    for data, labels in train_loader:
        data = data.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        transform = v2.Compose([
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.RandomRotation(degrees=(-30, 30)),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            v2.RandomResizedCrop(size=(28, 28), scale=(0.8, 1.0), ratio=(0.8, 1.2)),
            v2.ToTensor(),
        ])
        data = transform(data)
        output = model(data)
        loss = criterion(output, labels)

        loss.backward()

        # torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        output = output.softmax(dim=1).detach().cpu().squeeze()
        labels = labels.cpu().squeeze()
        all_outputs.append(output)
        all_labels.append(labels)

    all_outputs = torch.cat(all_outputs).argmax(dim=1)
    all_labels = torch.cat(all_labels)

    return round(accuracy(all_outputs, all_labels), 4)


def val(model, val_loader, device):
    model.eval()

    all_outputs = []
    all_labels = []

    for data, labels in val_loader:
        data = data.to(device, non_blocking=True)

        with torch.no_grad():
            output = model(data)

        output = output.softmax(dim=1).cpu().squeeze()
        labels = labels.squeeze()
        all_outputs.append(output)
        all_labels.append(labels)

    all_outputs = torch.cat(all_outputs).argmax(dim=1)
    all_labels = torch.cat(all_labels)

    return round(accuracy(all_outputs, all_labels), 4)


def do_epoch(model, train_loader, val_loader, criterion, optimizer, device):
    acc = train(model, train_loader, criterion, optimizer, device)
    acc_val = val(model, val_loader, device)
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
        v2.ToTensor(),
        v2.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
    ]

    data_path = '../data'
    train_dataset = CIFAR10(root=data_path, train=True, transform=v2.Compose(transforms), download=True)
    val_dataset = CIFAR10(root=data_path, train=False, transform=v2.Compose(transforms), download=True)
    train_dataset = TrainCachedDataset(train_dataset)
    val_dataset = CachedDataset(val_dataset)

    model = ConvNet(3, 28, 10)
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    epochs = 50

    batch_size = 1000
    val_batch_size = 1000
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
        acc, acc_val = do_epoch(model, train_loader, val_loader, criterion, optimizer, device)
        tbar.set_postfix_str(f"Acc: {acc}, Acc_val: {acc_val}")
        writer.add_scalar("Train/Accuracy", acc, epoch)
        writer.add_scalar("Val/Accuracy", acc_val, epoch)
        writer.add_scalar("Model/Norm", get_model_norm(model), epoch)


if __name__ == '__main__':
    freeze_support()
    main()
