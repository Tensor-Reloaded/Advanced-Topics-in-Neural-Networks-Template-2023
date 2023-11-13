import torch
import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import v2


class CIFAR10DataLoadersBuilder:
    def __init__(self, seed, dataset_path, train_ratio=0.8, batch_size=32, device=None):
        self.loaded_dataset = None
        self.test_loader = None
        self.train_loader = None
        self.train_set = None
        self.test_set = None
        self.train_ratio = train_ratio
        self.batch_size = batch_size
        self.device_type = device
        self.seed = seed
        self.dataset_path = dataset_path
        self.transforms = [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((28, 28), antialias=True),
            v2.Grayscale(),
            torch.flatten,
        ]
        self.dataset_loaded = False

    def load_datasets(self):
        if self.dataset_loaded:
            return self.loaded_dataset

        transform = torchvision.transforms.Compose(self.transforms)
        self.loaded_dataset = torchvision.datasets.CIFAR10(root=self.dataset_path, train=True, download=True,
                                                    transform=transform)


        self.dataset_loaded = True

        return self.train_set, self.test_set

    def get_data_loaders(self):
        if self.train_loader is not None and self.test_loader is not None:
            return self.train_loader, self.test_loader

        self.load_datasets()

        train_size = int(self.train_ratio * len(self.loaded_dataset))
        test_size = len(self.loaded_dataset) - train_size

        train_dataset, test_dataset = random_split(self.loaded_dataset, [train_size, test_size])
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size)

        return self.train_loader, self.test_loader

