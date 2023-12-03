#!/usr/bin/env python3
import argparse
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import v2
import matplotlib.pyplot as plt
from nn.util.device import get_default_device
from nn.model.model import Model
from nn.dataset.custom_dataset import CustomDataset
from util.args import parse_arguments

current_path = os.path.dirname(__file__)


def main():
    args = parse_arguments(require_weights=True)
    device = get_default_device()
    weights_path = get_weights_path(args)

    dataset = CustomDataset(
        train=False, cache=False, data_path=f"{current_path}/../data/datasets"
    )
    dataset = torch.stack(dataset.images)
    dataset = TensorDataset(dataset)
    dataloader = DataLoader(
        dataset, batch_size=100, num_workers=2, pin_memory=device == "cuda"
    )
    ground_truth_transforms = v2.Compose(
        [
            v2.Resize((28, 28), antialias=True),
            v2.Grayscale(),
            v2.functional.hflip,
            v2.functional.vflip,
        ]
    )
    model = Model(device=device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.zero_grad()
    model.eval()

    for images in dataloader:
        x = images[0]
        y = ground_truth_transforms(x)
        y_hat = model(x=x)

        for image in range(x.shape[0]):
            plt.figure(figsize=(12, 4))

            plt.subplot(1, 3, 1)
            plt.imshow(x[image].permute(1, 2, 0))
            plt.title("Original image")

            plt.subplot(1, 3, 2)
            plt.imshow(y[image].permute(1, 2, 0).detach().numpy(), cmap="gray")
            plt.title("Ground truth image")

            plt.subplot(1, 3, 3)
            plt.imshow(y_hat[image].permute(1, 2, 0).detach().numpy(), cmap="gray")
            plt.title("Generated image")

            plt.savefig(f"{current_path}/../data/outputs/output_{image}.png")
            plt.show()


def get_weights_path(args: argparse.Namespace):
    path = f"{current_path}/../{args.weights}"

    if not os.path.exists(path):
        raise FileNotFoundError(f"Weights path not found at {path}")

    return path


if __name__ == "__main__":
    main()
