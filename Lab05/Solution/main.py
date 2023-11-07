import os
from multiprocessing import freeze_support

import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2

from trainingPipeline import *
from models import *
import wandb


# TODO:General Cleanup


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
        # For multi-gpu workstations, PyTorch will use the first available GPU (cuda:0), unless specified otherwise
        # (cuda:1).
    if torch.backends.mps.is_available():
        return torch.device('mos')
    return torch.device('cpu')


def run_model(device=get_default_device()):
    size = (28, 28)

    transforms = [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize(size, antialias=True),
        v2.Grayscale(),
        v2.Normalize((0.5,), (0.5,), inplace=True),
    ]

    data_path = '../data'
    train_dataset = CIFAR10(root=data_path, train=True, transform=v2.Compose(transforms), download=True)
    validation_dataset = CIFAR10(root=data_path, train=False, transform=v2.Compose(transforms), download=True)

    no_units_per_layer = [784, 128, 64, 10]
    model = MLP(device, no_units_per_layer)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.026172624468404335, momentum=0.01964499304214733,
    #                             weight_decay=0.090403235101392, nesterov=True)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    no_epochs = 50

    # TODO:Add data augmentation

    train_batch_size = 256
    validation_batch_size = 500
    num_workers = 2
    train_transforms = None
    train_transforms = v2.Compose([
        v2.RandomHorizontalFlip(),
        v2.GaussianBlur(3, 0.1),
        torch.flatten
    ])

    val_transforms = None
    val_transforms = v2.Compose([
        torch.flatten
    ])

    training_pipeline = TrainingPipeline(device, False, train_dataset, validation_dataset,
                                         train_transformer=train_transforms, val_transformer=val_transforms,
                                         cache=True,
                                         train_batch_size=train_batch_size, val_batch_size=validation_batch_size,
                                         no_workers=num_workers)
    print("Device: ", device)
    training_pipeline.run(no_epochs, model, criterion, optimizer)


# TODO:Clean memory to ensure longer runs fight this lag
def run_sweep(device=get_default_device()):
    # Set up the pipeline
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

    train_transforms = None
    # train_transforms = v2.Compose([
    # ])

    training_pipeline = TrainingPipeline(device, use_config_for_train=True, train_dataset=train_dataset,
                                         val_dataset=None,
                                         train_transformer=train_transforms, cache=True,
                                         train_batch_size=None, val_batch_size=None,
                                         no_workers=8)

    # Step 1
    wandb.login()

    sweep_config = {
        'method': 'random'
    }

    metric = {
        'name': 'loss',
        'goal': 'minimize'
    }

    sweep_config['metric'] = metric

    parameters_dict = {
        'epochs': {
            'value': 3
        },
        'optimizer': {
            'value': 'sgd'
        },
        'no_units_per_layer': {
            'value': [784, 128, 64, 10]
        },
        'batch_size': {
            'values': [32, 64, 128, 256]
        },
        'learning_rate': {
            # a flat distribution between 0 and 0.1
            'distribution': 'uniform',
            'min': 0.0001,
            'max': 0.1
        },
        'momentum': {
            'distribution': 'uniform',
            'min': 0.0001,
            'max': 0.1
        },
        'weight_decay': {
            'distribution': 'uniform',
            'min': 0.001,
            'max': 0.5
        },
        'nesterov': {
            'value': True
        }
    }

    sweep_config['parameters'] = parameters_dict

    # Step 2
    sweep_id = wandb.sweep(sweep_config, project="H5 SGD Hyper-parameters tuning")

    # Step 3
    wandb.agent(sweep_id, training_pipeline.run_config, count=40)


if __name__ == '__main__':
    freeze_support()
    run_model()

    # run_sweep()

# python -m tensorboard.main --logdir=runs
# this commands works for me on Windows 10
