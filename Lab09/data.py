import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2


def get_cifar10_images(data_path: str, train: bool):
    initial_transforms = v2.Compose([
        v2.ToTensor(),
        v2.ConvertImageDtype(torch.float32)
    ])
    cifar_10_dataset = CIFAR10(root=data_path, train=train, transform=initial_transforms, download=True)
    return [(image, label) for image, label in cifar_10_dataset]


class CustomDataset(Dataset):
    def __init__(self, data_path: str = './data', train: bool = True, transform=None):
        self.dataset = get_cifar10_images(data_path, train)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]  # Ignore the label
        transformed_img = self.transform(img) if self.transform else img
        return img, transformed_img
