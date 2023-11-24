import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader


def get_cifar_10_loaders(config: dict) -> tuple[DataLoader, DataLoader]:
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2471, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(config["scale"], 1.0), ratio=(1.0, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandAugment(num_ops=config["ra_n"], magnitude=config["ra_m"]),
        transforms.ColorJitter(config["jitter"], config["jitter"], config["jitter"]),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
        transforms.RandomErasing(p=config["reprob"])
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std)
    ])

    train_set = CIFAR10(root='./data', train=True,
                        download=True, transform=train_transform)
    train_loader = DataLoader(train_set, batch_size=config["batch_size"],
                              shuffle=True, num_workers=config["workers"])

    test_set = CIFAR10(root='./data', train=False,
                       download=True, transform=test_transform)
    val_loader = DataLoader(test_set, batch_size=config["batch_size"],
                            shuffle=False, num_workers=config["workers"])

    return train_loader, val_loader

