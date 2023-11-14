import torch
import wandb
from torchvision.transforms import v2
from torchvision.datasets import CIFAR10
from multiprocessing import freeze_support
from torch.utils.data import DataLoader

from Model import MLP
from Config import Config
from Pipeline import Pipeline
from CachedDataset import CachedDataset
from sam import SAM

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    # if torch.backends.mps.is_available():
    #     return torch.device('mps')

    return torch.device('cpu')


def build_model(config):
    wandb.init(project="atnn-2023-lab5", config={
        "learning_rate": config.learning_rate,
        "dataset": "CIFAR-10",
        "epochs": config.epochs
    })
    model = MLP(784, 10, config)
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

    data_path = '../data'
    train_dataset = CIFAR10(root=data_path, train=True, transform=transforms, download=True)
    val_dataset = CIFAR10(root=data_path, train=False, transform=transforms, download=True)
    train_dataset = CachedDataset(train_dataset)
    val_dataset = CachedDataset(val_dataset)

    epochs = 50
    batch_size = 256
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
        Config(device, epochs, criterion, torch.optim.SGD, 0.05, 'SGD: lr = 0.05', './runs/SGD_0.05'),
        Config(device, epochs, criterion, torch.optim.SGD, 0.1, 'SGD: lr = 0.1', './runs/SGD_0.1'),
        Config(device, epochs, criterion, torch.optim.SGD, 0.15, 'SGD: lr = 0.15', './runs/SGD_0.15'),

        Config(device, epochs, criterion, torch.optim.Adam, 0.0001, 'Adam: lr = 0.0001', './runs/Adam_0.0001'),
        Config(device, epochs, criterion, torch.optim.Adam, 0.0005, 'Adam: lr = 0.0005', './runs/Adam_0.0005'),
        Config(device, epochs, criterion, torch.optim.Adam, 0.001, 'Adam: lr = 0.001', './runs/Adam_0.001'),

        Config(device, epochs, criterion, torch.optim.RMSprop, 0.0001, 'RMSprop: lr = 0.0001', './runs/RMSprop_0.0001'),
        Config(device, epochs, criterion, torch.optim.RMSprop, 0.001, 'RMSprop: lr = 0.001', './runs/RMSprop_0.001'),
        Config(device, epochs, criterion, torch.optim.RMSprop, 0.005, 'RMSprop: lr = 0.01', './runs/RMSprop_0.01'),

        Config(device, epochs, criterion, torch.optim.Adagrad, 0.005, 'Adagrad: lr = 0.001', './runs/Adagrad_0.001'),
        Config(device, epochs, criterion, torch.optim.Adagrad, 0.01, 'Adagrad: lr = 0.01', './runs/Adagrad_0.01'),
        Config(device, epochs, criterion, torch.optim.Adagrad, 0.05, 'Adagrad: lr = 0.05', './runs/Adagrad_0.05'),

        Config(device, epochs, criterion, SAM, 0.1, 'SAM with SGD: lr = 0.1', './runs/SAM_SGD_0.1', torch.optim.SGD),
        Config(device, epochs, criterion, SAM, 0.15, 'SAM with SGD: lr = 0.15', './runs/SAM_SGD_0.15', torch.optim.SGD),
        Config(device, epochs, criterion, SAM, 0.2, 'SAM with SGD: lr = 0.2', './runs/SAM_SGD_0.2', torch.optim.SGD)
    ]

    for config in model_configs:
        model = build_model(config)
        Pipeline.run(model, train_loader, val_loader, config)
        wandb.finish()

if __name__ == '__main__':
    freeze_support()
    main()