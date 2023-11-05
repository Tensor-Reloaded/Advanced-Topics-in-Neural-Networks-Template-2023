import torch
from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import v2
from torchvision.datasets import CIFAR10
import matplotlib.pyplot as plt
import numpy as np

__all__ = ['CachedDataset']


class CachedDataset(Dataset):
    def __init__(self, given_dataset: Dataset, transformer=None, cache: bool = True):
        self.transformers = transformer

        if cache:
            given_dataset = tuple([x for x in given_dataset])

        self.dataset = given_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        features, label = self.dataset[idx]
        if self.transformers is not None:
            features = self.transformers(features)
        return features, label


def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__ == '__main__':
    transforms = [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((28, 28), antialias=True),
        v2.Grayscale(),
        # torch.flatten,
    ]

    data_path = '../data'
    val_dataset = CIFAR10(root=data_path, train=False, transform=v2.Compose(transforms), download=True)
    val_dataset = CachedDataset(val_dataset, cache=True)

    batch_size = 4

    loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                         shuffle=False)

    dataiter = iter(loader)
    images, labels = next(dataiter)

    imshow(torchvision.utils.make_grid(images))

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))
