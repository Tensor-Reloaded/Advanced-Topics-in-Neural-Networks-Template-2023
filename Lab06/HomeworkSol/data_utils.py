import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataset import random_split
import random
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image


# Define your transformations and data loading functions here
class AdvancedAugmentations:
    def __init__(self):
        # Define individual transforms here
        self.color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        # self.cutout = transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            raise TypeError(f"Expected img to be a PIL Image, got {type(img)}")

        # Get width and height of the image
        width, height = img.size

        # Apply Color Jitter
        img = self.color_jitter(img)

        # Random Crop and Pad
        img = TF.pad(img, [4], padding_mode='reflect')
        img = TF.crop(img, top=random.randint(0, 8), left=random.randint(0, 8), height=32, width=32)

        # Cutout
        # img = self.cutout(img)

        # Random Horizontal Flip
        if random.random() > 0.5:
            img = TF.hflip(img)

        # Random Rotation
        if random.random() > 0.5:
            degrees = random.uniform(-10, 10)  # Rotating between -10 and 10 degrees
            img = TF.rotate(img, degrees)

        # Random Zoom
        if random.random() > 0.5:
            scale = random.uniform(0.8, 1.2)  # Zooming between 80% (zoom out) and 120% (zoom in)
            img = TF.affine(img, angle=0, translate=(0, 0), scale=scale, shear=0)

        # Random Shear
        if random.random() > 0.5:
            shear = random.uniform(-10, 10)
            img = TF.affine(img, angle=0, translate=(0, 0), scale=1.0, shear=shear)

        # Random Translation
        if random.random() > 0.5:
            max_dx = img.size[0] * 0.1
            max_dy = img.size[1] * 0.1
            translations = (np.round(random.uniform(-max_dx, max_dx)),
                            np.round(random.uniform(-max_dy, max_dy)))
            img = TF.affine(img, angle=0, translate=translations, scale=1.0, shear=0)

        # Further augmentations like MixUp or CutMix are applied during the batch preparation

        return img

def get_transforms():
    transform_train = transforms.Compose([
        AdvancedAugmentations(),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_valid = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return transform_train, transform_valid


def get_datasets(dataset_name, transform_train, transform_valid, root='./data'):
    if dataset_name == 'CIFAR10':
        trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform_valid)
    elif dataset_name == 'CIFAR100':
        trainset = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=transform_valid)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    validation_split = 0.1
    train_size = int((1 - validation_split) * len(trainset))
    validation_size = len(trainset) - train_size

    train_dataset, validation_dataset = random_split(trainset, [train_size, validation_size])

    return train_dataset, validation_dataset, testset


def get_dataloaders(train_dataset, validation_dataset, testset, batch_size=128, num_workers=2):
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                              num_workers=num_workers)
    validloader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False,
                                              num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, validloader, testloader


def prepare_data_loaders(config):
    # Use the config to adjust any data parameters
    # For example: config['batch_size'] or config['dataset_root']
    transform_train, transform_valid = get_transforms()
    train_dataset, validation_dataset, testset = get_datasets(config['dataset_name'], transform_train, transform_valid,
                                                              root=config['dataset_root'])
    trainloader, validloader, testloader = get_dataloaders(train_dataset, validation_dataset, testset,
                                                           batch_size=config['batch_size'],
                                                           num_workers=config['num_workers'])

    return trainloader, validloader, testloader

