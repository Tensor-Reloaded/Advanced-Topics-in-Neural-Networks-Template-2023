import gc
from functools import wraps
from multiprocessing import freeze_support
from time import time

import torch
from torch import nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader, TensorDataset


def timed(fn: callable):
    @wraps(fn)
    def wrap(*args, **kwargs):
        gc.collect()
        start = time()
        fn(*args, **kwargs)
        end = time()
        return end - start

    return wrap


def get_cifar10_images(data_path: str, train: bool):
    initial_transforms = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    cifar_10_images = CIFAR10(root=data_path, train=train, transform=initial_transforms, download=True)
    return [image for image, label in cifar_10_images]


class CustomDataset(Dataset):
    def __init__(self, data_path: str = './data', train: bool = True, cache: bool = True):
        self.images = get_cifar10_images(data_path, train)
        self.cache = cache
        self.transforms = v2.Compose([
            v2.Resize((28, 28), antialias=True),
            v2.Grayscale(),
            v2.functional.hflip,
            v2.functional.vflip,
        ])
        if cache:
            self.labels = [self.transforms(x) for x in self.images]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        if self.cache:
            return self.images[i], self.labels[i]
        return self.images[i], self.transforms(self.images[i])


@timed
def transform_dataset_with_transforms(dataset: TensorDataset):
    transforms = v2.Compose([
        v2.Resize((28, 28), antialias=True),
        v2.Grayscale(),
        v2.functional.hflip,
        v2.functional.vflip,
    ])
    for image in dataset.tensors[0]:
        transforms(image)


@timed
@torch.no_grad()
def transform_dataset_with_model(dataset: TensorDataset, model: nn.Module, batch_size: int):
    # model.eval()  # TODO: uncomment this
    dataloader = DataLoader(dataset, batch_size=batch_size)  # TODO: Complete the other parameters
    for images in dataloader:
        # model(images)  # TODO: uncomment this
        pass


def test_inference_time(model: nn.Module, device=torch.device('cpu')):
    test_dataset = CustomDataset(train=False, cache=False)
    test_dataset = torch.stack(test_dataset.images)
    test_dataset = TensorDataset(test_dataset)

    batch_size = 100  # TODO: add the other parameters (device, ...)

    t1 = transform_dataset_with_transforms(test_dataset)
    t2 = transform_dataset_with_model(test_dataset, model, batch_size)
    print(f"Sequential transforming each image took: {t1} on CPU. \n"
          f"Using a model with batch_size: {batch_size} took {t2} on {device.type}. \n")


def main():
    test_inference_time(None)


if __name__ == '__main__':
    freeze_support()
    main()
