import numpy as np
import torch
import torch.optim as optim
from torch import nn
from conv_mixer_network import ConvMixer
from torch.utils.data import DataLoader
from trainer import train_epochs
from wba_manager import WBAManager
from dataset import get_cifar_10_loaders
from utils import get_default_device


def main():
    manager = WBAManager()
    config = manager.get_config()
    train_loader, val_loader = get_cifar_10_loaders(config)

    model = ConvMixer(config["hdim"], config["depth"], patch_size=config["psize"], kernel_size=config["conv_ks"], n_classes=10)
    model = nn.DataParallel(model).cuda()
    device = get_default_device()

    lr_schedule = lambda t: np.interp([t], [0, config["epochs"] * 2 // 5, config["epochs"] * 4 // 5, config["epochs"]],
                                      [0, config["lr_max"], config["lr_max"] / 20.0, 0])[0]

    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    optimizer = optim.AdamW(model.parameters(), lr=config["lr_max"], weight_decay=config["wd"])

    train_epochs(manager, manager.config["epochs"], model, train_loader, val_loader, criterion, optimizer,
                 manager.config["optimizer_name"],
                 manager.config["batch_size"], scaler, config, lr_schedule, device)


if __name__ == "__main__":
    main()

