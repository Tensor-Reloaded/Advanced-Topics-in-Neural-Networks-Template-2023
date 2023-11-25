import torch
from torch import nn
from conv_mixer_network import ConvMixer
from wba_manager import WBAManager
from dataset import get_cifar_10_loaders
from trainer import val
from utils import get_default_device


def main():
    manager = WBAManager()
    config = manager.get_config()
    train_loader, val_loader = get_cifar_10_loaders(config)

    model = ConvMixer(config["hdim"], config["depth"], patch_size=config["psize"], kernel_size=config["conv_ks"],
                      n_classes=10)

    model.load_state_dict(torch.load("./saved_model"))

    device = get_default_device()

    criterion = nn.CrossEntropyLoss()

    val_acc, val_loss = val(model, val_loader, criterion, device)

    print(f"Accuracy: {val_acc}")


if __name__ == "__main__":
    main()
