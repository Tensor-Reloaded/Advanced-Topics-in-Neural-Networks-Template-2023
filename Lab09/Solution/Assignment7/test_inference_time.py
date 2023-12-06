from time import time

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2

from Assignment7.dataset import Dataset
from Assignment7.utils import build_transformed_dataset


def transform_dataset_with_transforms(dataset: Dataset):
    start = time()
    transforms = v2.Compose([
        v2.Resize((28, 28), antialias=True),
        v2.Grayscale(),
        v2.functional.hflip,
        v2.functional.vflip,
    ])
    for image in dataset:
        transforms(image[0])
    end = time()
    return end - start


def transform_dataset_with_model(val_dataset, model: torch.nn.Module, batch_size: int,
                                 device=torch.device('cpu')):
    start = time()
    val_loader = DataLoader(val_dataset, shuffle=False, pin_memory=True, num_workers=0,
                            batch_size=batch_size, drop_last=False)
    model.eval()  # TODO: uncomment this
    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(device)
            outputs = model(features)
    end = time()
    return end - start


def test_inference_time(model: torch.nn.Module, device=torch.device('cpu')):
    transforms_test = [v2.ToImage(),
                       v2.ToDtype(torch.float32, scale=True)]
    val_dataset = CIFAR10(root='../data', train=False,
                          transform=v2.Compose(transforms_test), download=True)
    val_dataset = Dataset(val_dataset,
                          build_transformed_dataset,
                          transformations=[], transformations_test=[], training=False,
                          save=False)
    batch_size = 100
    t1 = transform_dataset_with_transforms(val_dataset)
    t2 = transform_dataset_with_model(val_dataset, model, batch_size=100, device=device)
    print(f"Sequential transforming each image took: {t1} on CPU. \n"
          f"Using a model with batch_size: {batch_size} took {t2} on {device.type}. \n")
    return t1, t2
