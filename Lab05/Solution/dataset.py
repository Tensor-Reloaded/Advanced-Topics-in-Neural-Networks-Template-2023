import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2


class Cifar10Dataset(Dataset):
    def __init__(self, dataset, cache=True):
        if cache:
            dataset = tuple([f for f in dataset])
        self.dataset = dataset

        self.transforms = [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((28, 28), antialias=True),
            v2.Grayscale(),
            torch.flatten,
        ]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]