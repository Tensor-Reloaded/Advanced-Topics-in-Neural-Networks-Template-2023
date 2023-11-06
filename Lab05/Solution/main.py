import os
from multiprocessing import freeze_support

import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2

from trainingPipeline import *
from models import *


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
        # For multi-gpu workstations, PyTorch will use the first available GPU (cuda:0), unless specified otherwise
        # (cuda:1).
    if torch.backends.mps.is_available():
        return torch.device('mos')
    return torch.device('cpu')


def main(device=get_default_device()):
    size = (28, 28)

    transforms = [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize(size, antialias=True),
        v2.Grayscale(),
        v2.Normalize((0.5,), (0.5,), inplace=True),
        torch.flatten,
    ]

    data_path = '../data'
    train_dataset = CIFAR10(root=data_path, train=True, transform=v2.Compose(transforms), download=True)
    validation_dataset = CIFAR10(root=data_path, train=False, transform=v2.Compose(transforms), download=True)

    no_units_per_layer = [784, 128, 64, 10]
    model = MLP(device, no_units_per_layer)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.001, weight_decay=0.05, nesterov=True)
    criterion = torch.nn.CrossEntropyLoss()
    no_epochs = 50

    # TODO:Add data augmentation

    train_batch_size = 256
    validation_batch_size = 500
    num_workers = 2
    train_transforms = None
    # train_transforms = v2.Compose([
    # ])

    training_pipeline = TrainingPipeline(device, train_dataset, validation_dataset,
                                         train_transformers=train_transforms, cache=True,
                                         train_batch_size=train_batch_size, validation_batch_size=validation_batch_size,
                                         no_workers=num_workers)
    print("Device: ", device)
    training_pipeline.run(no_epochs, model, criterion, optimizer)


if __name__ == '__main__':
    freeze_support()
    main()

# python -m tensorboard.main --logdir=runs
# this commands works for me on Windows 10
