import os
from multiprocessing import freeze_support

from torch.utils.data import DataLoader

from sam import SAM

import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2
from datasets import *

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


def accuracy_and_loss(device: torch.device, model, data_loader, criterion) -> tuple[float, float]:
    model.eval()

    all_outputs = []
    all_labels = []

    total_loss = 0.0

    for data, labels in data_loader:
        data = data.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.no_grad():
            output = model(data)

        loss = criterion(output, labels)
        total_loss += loss.item()

        output = output.softmax(dim=1).detach().cpu().squeeze()
        labels = labels.cpu().squeeze()
        all_outputs.append(output)
        all_labels.append(labels)

    all_outputs = torch.cat(all_outputs).argmax(dim=1).to(device, non_blocking=True)
    all_labels = torch.cat(all_labels).to(device, non_blocking=True)

    fp_plus_fn = torch.logical_not(all_outputs == all_labels).sum().item()
    all_elements = all_outputs.shape[0]

    return (all_elements - fp_plus_fn) / all_elements, total_loss


def main(device: torch.device = get_default_device()):
    transforms = [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]

    data_path = '../data'
    validation_dataset = CIFAR10(root=data_path, train=False, transform=v2.Compose(transforms), download=True)

    validation_dataset = CachedDataset(validation_dataset, None, True)
    validation_loader = DataLoader(validation_dataset, shuffle=False, pin_memory=True, num_workers=0,
                                   batch_size=500, drop_last=False)

    model_path = "model.pth"
    model = CNN(device, 10)
    model.load_state_dict(torch.load(model_path))

    criterion = torch.nn.CrossEntropyLoss()

    val_accuracy, val_loss = accuracy_and_loss(device, model, validation_loader, criterion)

    print("Validation Accuracy : ", val_accuracy)
    print("Validation Loss: ", val_loss)


if __name__ == '__main__':
    main()
