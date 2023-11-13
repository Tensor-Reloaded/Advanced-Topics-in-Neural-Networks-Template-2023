from multiprocessing import freeze_support

import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from CachedDataset import CachedDataset
from Config import Config
from Model import Model
from Pipeline import Pipeline
from SAM import SAM
import wandb


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    # if torch.backends.mps.is_available():
    #     return torch.device('mos')
    return torch.device('cpu')


def build_model(config):
    wandb.init(project="atnn_homework5", config={
        "learning_rate": config.learning_rate,
        "dataset": "CIFAR-10",
        "epochs": config.epochs
    })
    model = Model(784, 10, config)
    model = model.to(config.device)
    wandb.watch(models=model, criterion=config.criterion)

    return model


def main(device=get_default_device()):
    transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((28, 28), antialias=True),
        v2.Grayscale(),
        torch.flatten,
    ])
    train_transforms = v2.Compose([
        v2.ToImage(),
        v2.RandAugment(),
        v2.RandomHorizontalFlip(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((28, 28), antialias=True),
        v2.Grayscale(),
        torch.flatten,
    ])
    rand_transforms = v2.Compose([
        v2.RandAugment(),
        v2.Grayscale(),
        v2.Normalize([0.4733], [0.2515]),
        torch.flatten,
    ])

    data_path = '../data'
    train_dataset = CIFAR10(root=data_path, train=True, transform=transforms, download=True)
    val_dataset = CIFAR10(root=data_path, train=False, transform=transforms, download=True)
    train_dataset = CachedDataset(train_dataset)
    val_dataset = CachedDataset(val_dataset)

    batch_size = 512
    val_batch_size = 500
    num_workers = 2
    persistent_workers = (num_workers != 0)
    pin_memory = device.type == 'cuda'
    train_loader = DataLoader(train_dataset, shuffle=True, pin_memory=pin_memory, num_workers=num_workers,
                              batch_size=batch_size, drop_last=True, persistent_workers=persistent_workers)
    val_loader = DataLoader(val_dataset, shuffle=False, pin_memory=True, num_workers=0, batch_size=val_batch_size,
                            drop_last=False)

    criterion = torch.nn.CrossEntropyLoss()
    model_configs = [
        Config(device, 150, criterion, SAM, 0.17, 'SAM: lr = 0.17', r'./runs/SAM_1', torch.optim.SGD),
        Config(device, 100, criterion, SAM, 0.01, 'SAM: lr = 0.01', r'./runs/SAM_2', torch.optim.SGD),
        Config(device, 100, criterion, SAM, 0.001, 'SAM: lr = 0.001', r'./runs/SAM_3', torch.optim.SGD),
        Config(device, 100, criterion, torch.optim.SGD, 0.1, 'SGD: lr = 0.1', r'./runs/SGD_1', momentum=0.9),
        Config(device, 100, criterion, torch.optim.SGD, 0.05, 'SGD: lr = 0.05', r'./runs/SGD_2', momentum=0.9),
        Config(device, 100, criterion, torch.optim.SGD, 0.15, 'SGD: lr = 0.15', r'./runs/SGD_3', momentum=0.9),
        Config(device, 50, criterion, torch.optim.Adam, 0.01, 'Adam: lr = 0.01', r'./runs/Adam_1',
               betas=(0.9, 0.999), eps=1e-8),
        Config(device, 50, criterion, torch.optim.Adam, 0.005, 'Adam: lr = 0.005', r'./runs/Adam_2',
               betas=(0.9, 0.999), eps=1e-8),
        Config(device, 50, criterion, torch.optim.Adam, 0.001, 'Adam: lr = 0.001', r'./runs/Adam_3',
               betas=(0.9, 0.999), eps=1e-8),
        Config(device, 50, criterion, torch.optim.RMSprop, 0.0001, 'RMSprop: lr = 0.0001', r'./runs/RMSprop_1',
               alpha=0.9),
        Config(device, 50, criterion, torch.optim.RMSprop, 0.0002, 'RMSprop: lr = 0.0002', r'./runs/RMSprop_2',
               alpha=0.9),
        Config(device, 50, criterion, torch.optim.RMSprop, 0.0005, 'RMSprop: lr = 0.0005', r'./runs/RMSprop_3',
               alpha=0.9),
        Config(device, 50, criterion, torch.optim.Adagrad, 0.001, 'Adagrad: lr = 0.001', r'./runs/Adagrad_1',
               lr_decay=0.9),
        Config(device, 50, criterion, torch.optim.Adagrad, 0.005, 'Adagrad: lr = 0.005', r'./runs/Adagrad_2',
               lr_decay=0.9),
        Config(device, 50, criterion, torch.optim.Adagrad, 0.01, 'Adagrad: lr = 0.01', r'./runs/Adagrad_3',
               lr_decay=0.9),
       ]

    for config in model_configs:
        model = build_model(config)
        Pipeline.run(model, train_loader, val_loader, config)


def main_wrapper():
    sweep_config = {
        'method': 'random',
        'metric': {'goal': 'maximize', 'name': 'accuracy'},
        'parameters': {
            'learning_rate': {'min': 0.0001, 'max': 0.1},
            'batch_size': {'values': [32, 64, 128]},
            'num_epochs': {'values': [50, 100, 150]}
        }
    }

    sweep_id = wandb.sweep(sweep_config, project='ATNN_Homework5')
    wandb.agent(sweep_id, function=main)


if __name__ == '__main__':
    freeze_support()
    main_wrapper()
