import os
from multiprocessing import freeze_support
from sam import SAM

import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2

from trainingPipeline import *
from models import *
import wandb


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
        # For multi-gpu workstations, PyTorch will use the first available GPU (cuda:0), unless specified otherwise
        # (cuda:1).
    if torch.backends.mps.is_available():
        return torch.device('mos')
    return torch.device('cpu')


def get_optimizer(model: torch.nn.Module, optimizer_name: str, optimizer_params: dict = None) -> torch.optim:
    optimizer = None
    if optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), **optimizer_params)
    elif optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), **optimizer_params)
    elif optimizer_name == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), **optimizer_params)
    elif optimizer_name == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), **optimizer_params)
    elif optimizer_name == 'sam':
        # TODO:Also try SAM with other base optimizers
        base_optimizer = torch.optim.SGD
        optimizer = SAM(model.parameters(), base_optimizer, **optimizer_params)

    return optimizer


def run_model(optimizer_name: str, optimizer_params: dict = None, train_batch_size: int = None,
              device: torch.device = get_default_device()):
    transforms = [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]

    # (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)

    data_path = '../data'
    train_dataset = CIFAR10(root=data_path, train=True, transform=v2.Compose(transforms), download=True)
    validation_dataset = CIFAR10(root=data_path, train=False, transform=v2.Compose(transforms), download=True)

    no_epochs = 5

    model = CNN(device, no_classes=10)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.026172624468404335, momentum=0.01964499304214733,
    #                             weight_decay=0.090403235101392, nesterov=True)

    optimizer = get_optimizer(model, optimizer_name, optimizer_params)

    criterion = torch.nn.CrossEntropyLoss()

    train_batch_size = 128
    validation_batch_size = 500
    num_workers = 2
    train_transforms = None
    train_transforms = v2.Compose([
        # v2.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        v2.RandAugment()
        # v2.AutoAugment(policy=v2.AutoAugmentPolicy.CIFAR10),
    ])

    val_transforms = None
    # val_transforms = v2.Compose([
    #     v2.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    # ])

    training_pipeline = TrainingPipeline(device, False, train_dataset, validation_dataset,
                                         train_transformer=train_transforms, val_transformer=val_transforms,
                                         cache=True,
                                         train_batch_size=train_batch_size, val_batch_size=validation_batch_size,
                                         no_workers=num_workers)
    print("Device: ", device)
    training_pipeline.run(no_epochs, model, criterion, optimizer)

    # torch.save(training_pipeline.model.state_dict(), "model.pth")


def get_sweep_params(optimizer_name: str) -> dict:
    parameters_dict = {
        'epochs': {
            'value': 10
        },
        'model': {
            'value': {
                'no_units_per_layer': [784, 512, 256, 128, 64, 10],
                'dropout_per_layer': [0.15, 0.15, 0.15, 0.15, 0.15],
                'skip_connections': []
            }
        },
        'batch_size': {
            'values': [32, 64, 128, 256]
        },
        'optimizer_name': {
            'value': optimizer_name
        }
    }

    if optimizer_name == 'SGD':
        parameters_dict.update({
            'lr': {
                'distribution': 'uniform',
                'min': 1e-4,
                'max': 1e-2
            },
            'momentum': {
                'distribution': 'uniform',
                'min': 1e-4,
                'max': 1e-1
            },
            'weight_decay': {
                'values': [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.0]
            },
            'nesterov': {
                'values': [False, True]
            },
        })
    elif optimizer_name == 'Adam':
        parameters_dict.update({
            'lr': {
                'distribution': 'uniform',
                'min': 1e-4,
                'max': 1e-2
            },
            'weight_decay': {
                'values': [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.0]
            }
        })
    elif optimizer_name == 'RMSprop':
        parameters_dict.update({
            'lr': {
                'distribution': 'uniform',
                'min': 1e-4,
                'max': 1e-2
            },
            'momentum': {
                'distribution': 'uniform',
                'min': 1e-4,
                'max': 1e-1
            },
            'weight_decay': {
                'values': [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.0]
            }
        })
    elif optimizer_name == 'Adagrad':
        parameters_dict.update({
            'lr': {
                'distribution': 'uniform',
                'min': 1e-4,
                'max': 1e-2
            },
            'lr_decay': {
                'distribution': 'uniform',
                'min': 0.9,
                'max': 1
            },
            'weight_decay': {
                'values': [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.0]
            }
        })
    elif optimizer_name == 'SAM with SGD':
        parameters_dict.update({
            'lr': {
                'distribution': 'uniform',
                'min': 1e-4,
                'max': 1e-2
            },
            'momentum': {
                'distribution': 'uniform',
                'min': 1e-4,
                'max': 1e-1
            },
            'weight_decay': {
                'values': [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.0]
            },
            'nesterov': {
                'values': [False, True]
            },
        })

    return parameters_dict


# TODO:Clean memory to ensure longer runs
def run_sweep(optimizer_name: str, sweep_id: str = None, device: torch.device = get_default_device()):
    # Set up the pipeline
    size = (28, 28)

    transforms = [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.AutoAugment(policy=v2.AutoAugmentPolicy.CIFAR10),
        # v2.Resize(size, antialias=True),
        # v2.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261), inplace=True),
        # v2.Grayscale(),
    ]

    data_path = '../data'
    train_dataset = CIFAR10(root=data_path, train=True, transform=v2.Compose(transforms), download=True)
    validation_dataset = CIFAR10(root=data_path, train=False, transform=v2.Compose(transforms), download=True)

    train_transforms = None
    train_transforms = v2.Compose([
        v2.RandomHorizontalFlip(),
        # v2.GaussianBlur(3),
        # torch.flatten
    ])
    val_transforms = v2.Compose([
        # torch.flatten
    ])

    training_pipeline = TrainingPipeline(device, use_config_for_train=True,
                                         train_dataset=train_dataset, val_dataset=validation_dataset,
                                         train_transformer=train_transforms, val_transformer=val_transforms,
                                         cache=True,
                                         train_batch_size=None, val_batch_size=None,
                                         no_workers=4)

    # Step 1
    wandb.login()

    sweep_config = {
        'method': 'random'
    }

    metric = {
        'name': 'validation_accuracy',
        'goal': 'maximize'
    }

    sweep_config['metric'] = metric

    sweep_config['parameters'] = get_sweep_params(optimizer_name)

    # Step 2
    project_name = "H5 " + optimizer_name + " Fine Tuning"
    if sweep_id is None:
        sweep_id = wandb.sweep(sweep_config, project=project_name)

    print("Project name: ", project_name)
    print("Sweep Id: ", sweep_id)

    # Step 3
    wandb.agent(sweep_id, training_pipeline.run_config, count=10)


if __name__ == '__main__':
    freeze_support()

    # These are the parameters for the best runs
    # run_model('sgd',
    #           {"lr": 0.00910, 'momentum': 0.0348287254025958, 'dampening': 0.0, 'weight_decay': 1e-5,
    #            'nesterov': True}, train_batch_size=32)
    #
    run_model('adam', {'lr': 1e-3, 'weight_decay': 0.00002})

    # run_model('rmsprop', {'lr': 0.00062, 'momentum': 0.006331417928678274, 'weight_decay': 0.0},
    #           train_batch_size=128)
    #
    # run_model('adagrad', {'lr': 0.00436, 'lr_decay': 0.95, 'weight_decay': 0.001}, train_batch_size=256)

    # run_model('sam', {"lr": 0.00865860767880207, 'momentum': 0.017413258272960617, 'dampening': 0.0, 'weight_decay': 0.0005,
    #         'nesterov': False})

    # run_sweep('RMSprop')

# python -m tensorboard.main --logdir=runs
# this commands works for me on Windows 10
