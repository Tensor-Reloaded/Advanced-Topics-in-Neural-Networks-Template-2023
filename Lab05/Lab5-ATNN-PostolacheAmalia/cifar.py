from multiprocessing import freeze_support
import wandb
import torch
from torchvision.datasets import CIFAR10
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
from dataset import *
from mlp import *
from training import *

wandb.init(
    project="lab5_cifar_10_pipeline",
    config={
        "seed": 300,
        "lr": 0.09,
        "dataset": "cifar-10",
        "epochNumber": 50,
    }
)

# sweep = wandb.sweep(wandb.config, project="lab5_cifar_10_pipeline")

def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
        # For multi-gpu workstations, PyTorch will use the first available GPU (cuda:0), unless specified otherwise
        # (cuda:1).
    if torch.backends.mps.is_available():
        return torch.device('mos')
    return torch.device('cpu')

def do_epoch(model, train_loader, val_loader, criterion, optimizer, device, epoch):
    acc, epoch_loss = train(model, train_loader, criterion, optimizer, epoch)
    acc_val, epoch_loss_val = val(model, val_loader, criterion, epoch)
    # torch.cuda.empty_cache()
    return acc, acc_val, epoch_loss, epoch_loss_val

def main(device=get_default_device()):
    transforms = [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((28, 28), antialias=True),
            v2.Grayscale(),
            torch.flatten,
        ]

    data_path = '../data'
    train_dataset = CIFAR10(root=data_path, train=True, transform=v2.Compose(transforms), download=True)
    val_dataset = CIFAR10(root=data_path, train=False, transform=v2.Compose(transforms), download=True)
    train_dataset = Cifar10Dataset(train_dataset)
    val_dataset = Cifar10Dataset(val_dataset)
    model = Cifar10MLP(784, 100, 10)
    model = model.to(device)

    config=wandb.config
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr)
    criterion = torch.nn.CrossEntropyLoss()
    epochs = 50

    batch_size = 128

    val_batch_size = 500
    num_workers = 2
    persistent_workers = (num_workers != 0)
    pin_memory = device.type == 'cuda'
    train_loader = DataLoader(train_dataset, shuffle=True, pin_memory=pin_memory, num_workers=num_workers,
                                batch_size=batch_size, drop_last=True, persistent_workers=persistent_workers)
    val_loader = DataLoader(val_dataset, shuffle=False, pin_memory=True, num_workers=0, batch_size=val_batch_size,
                            drop_last=False)

    writer = SummaryWriter()
    tbar = tqdm(tuple(range(epochs)))
    for epoch in tbar:
        acc, acc_val, epoch_loss, epoch_loss_val = do_epoch(model, train_loader, val_loader, criterion, optimizer, device, epoch)
        tbar.set_postfix_str(f"Acc: {acc}, Acc_val: {acc_val}")
        writer.add_scalar("Train/Accuracy", acc, epoch)
        writer.add_scalar("Val/Accuracy", acc_val, epoch)
        writer.add_scalar("Model/Norm", get_model_norm(model), epoch)

        writer.add_scalar("Train/EpochLoss", epoch_loss, epoch)
        writer.add_scalar("Val/EpochLoss", epoch_loss_val, epoch)

        writer.add_scalar("Hyperparameters/LearningRate", optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar("Hyperparameters/BatchSize", batch_size, epoch)
        writer.add_scalar("Hyperparameters/ValBatchSize", val_batch_size, epoch)

    writer.close()
    wandb.finish()
if __name__ == '__main__':
    freeze_support()
    main()