#!/usr/bin/env python3
import argparse
import os
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import v2
import matplotlib.pyplot as plt
from nn.util.device import get_default_device
from nn.model.model import Model
from nn.dataset.custom_dataset import CustomDataset
from util.util import timed
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

    test_inference_time(model=model, device=device)


def get_weights_path(args: argparse.Namespace):
    path = f"{current_path}/../{args.weights}"

    if not os.path.exists(path):
        raise FileNotFoundError(f"Weights path not found at {path}")

    return path


def test_inference_time(model: nn.Module, device=torch.device("cpu")):
    test_dataset = CustomDataset(
        train=False, cache=False, data_path=f"{current_path}/../data/datasets"
    )
    test_dataset = torch.stack(test_dataset.images)
    test_dataset = TensorDataset(test_dataset)

    batch_sizes = [1, 32, 64, 128, 256, 512, 1024]

    t1 = transform_dataset_with_transforms(test_dataset)
    print(f"Sequential transforming each image took: {t1} on CPU.")

    for batch_size in batch_sizes:
        t2 = transform_dataset_with_model(test_dataset, model, batch_size, device)
        print(
            f"Using a model with batch_size: {batch_size} took {t2} on {device.type}."
        )


@timed
def transform_dataset_with_transforms(dataset: TensorDataset):
    transforms = v2.Compose(
        [
            v2.Resize((28, 28), antialias=True),
            v2.Grayscale(),
            v2.functional.hflip,
            v2.functional.vflip,
        ]
    )
    for image in dataset.tensors[0]:
        transforms(image)


@timed
@torch.no_grad()
def transform_dataset_with_model(
    dataset: TensorDataset, model: nn.Module, batch_size: int, device: torch.device
):
    model.eval()
    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=2, pin_memory=device == "cuda"
    )
    for images in dataloader:
        model(x=images[0])


if __name__ == "__main__":
    main()
