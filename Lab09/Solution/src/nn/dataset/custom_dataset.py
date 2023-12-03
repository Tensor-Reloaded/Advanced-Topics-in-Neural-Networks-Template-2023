from torchvision.datasets import CIFAR10
from torchvision.transforms import v2, ToTensor
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(
        self, data_path: str = "./data", train: bool = True, cache: bool = True
    ):
        self.images = get_cifar10_images(data_path, train)
        self.cache = cache
        self.transforms = v2.Compose(
            [
                v2.Resize((28, 28), antialias=True),
                v2.Grayscale(),
                v2.functional.hflip,
                v2.functional.vflip,
            ]
        )
        if cache:
            self.labels = [self.transforms(x) for x in self.images]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        if self.cache:
            return self.images[i], self.labels[i]
        return self.images[i], self.transforms(self.images[i])


def get_cifar10_images(data_path: str, train: bool):
    initial_transforms = v2.Compose([ToTensor()])
    cifar_10_images = CIFAR10(
        root=data_path, train=train, transform=initial_transforms, download=True
    )
    return [image for image, label in cifar_10_images]
