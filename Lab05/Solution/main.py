import torch
from torchvision.transforms import v2
from dataset import CIFAR10Dataset
from simple_neural_network import SimpleNeuralNetwork
from compression_neural_network import CompressionNeuralNetwork
from upscale_neural_network import UpscaleNeuralNetwork
from torch.utils.data import DataLoader
from trainer import train_epochs
from wba_manager import WBAManager


def main(optimizer_name: str, config_idx: int):

    manager = WBAManager(optimizer_name, config_idx)

    #model, device = SimpleNeuralNetwork.for_device(784, 128, 10)
    #model, device = CompressionNeuralNetwork.for_device(128, 10)

    model, device = UpscaleNeuralNetwork().for_device(784, 128, 10)

    optimizer = manager.config["get_optimizer_fn"](model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    num_workers = 2
    persistent_workers = (num_workers != 0)
    pin_memory = device.type == 'cuda'
    train_loader = DataLoader(train_dataset, shuffle=True, pin_memory=pin_memory, num_workers=num_workers,
                              batch_size=manager.config["batch_size"], drop_last=True,
                              persistent_workers=persistent_workers)
    val_loader = DataLoader(val_dataset, shuffle=False, pin_memory=True, num_workers=0,
                            batch_size=manager.config["val_batch_size"], drop_last=False)

    train_epochs(manager, manager.config["epochs"], model, train_loader, val_loader, criterion, optimizer,
                 manager.config["optimizer_name"],
                 manager.config["batch_size"], device)


if __name__ == "__main__":

    transforms = [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((28, 28), antialias=True),
        v2.Grayscale(),
        torch.flatten,
    ]

    train_dataset = CIFAR10Dataset(True, transforms, True)
    val_dataset = CIFAR10Dataset(False, transforms, True)

    main("SGD", 0)
    main("SGD", 1)
    main("SGD", 2)

    main("Adam", 0)
    main("Adam", 1)
    main("Adam", 2)

    main("RMSprop", 0)
    main("RMSprop", 1)
    main("RMSprop", 2)

    main("Adagrad", 0)
    main("Adagrad", 1)
    main("Adagrad", 2)

    main("SGD_SAM", 0)
    main("SGD_SAM", 1)
    main("SGD_SAM", 2)


