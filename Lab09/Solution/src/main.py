#!/usr/bin/env python3
import os
import argparse
import typing as t
from multiprocessing import freeze_support

import torch
from torch import nn
from torchvision.transforms import v2
from torch.utils.data import DataLoader, TensorDataset
from nn.estimators.image_regression_accuracy_estimator import (
    ImageRegressionAccuracyEstimator,
)
from nn.model.model import Model
from nn.model.model_trainer import ModelTrainer
from nn.util.device import get_default_device
from nn.dataset.custom_dataset import CustomDataset
from util.util import timed

current_path = os.path.dirname(__file__)


def main():
    args = parse_arguments()
    device = get_default_device()

    init_model_strategy = get_model_init_strategy(args)
    model = init_model_strategy(device=device)
    test_inference_time(model=model, device=device)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-w",
        "--weights",
        required=False,
        help="Path to an already existing weights file",
    )
    args = parser.parse_args()

    return args


def get_model_init_strategy(args: argparse.Namespace):
    if args.weights is None:
        return train_model

    path = f"{current_path}/../{args.weights}"

    if not os.path.exists(path):
        return train_model

    return load_model(path)


def train_model(device: torch.device) -> nn.Module:
    training_dataset = CustomDataset(
        train=False, cache=False, data_path=f"{current_path}/../data/datasets"
    )
    validation_dataset = CustomDataset(
        train=True, cache=False, data_path=f"{current_path}/../data/datasets"
    )
    batched_train_dataset = DataLoader(
        dataset=training_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=2,
        pin_memory=device == "cuda",
    )
    batched_validation_dataset = DataLoader(
        dataset=validation_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=2,
        pin_memory=device == "cuda",
    )

    model = Model(device=device)
    model_trainer = ModelTrainer(
        model=model,
        loss_function=torch.nn.MSELoss,
        optimiser=torch.optim.Adam,
        learning_rate=0.001,
        accuracy_estimator=ImageRegressionAccuracyEstimator(precision=0.05),
        device=device,
        exports_path=f"{current_path}/../data/exports",
    )
    model_trainer.run(
        batched_training_dataset=batched_train_dataset,
        batched_validation_dataset=batched_validation_dataset,
        epochs=25,
    )

    model_trainer.export()

    return Model


def load_model(path: str) -> t.Callable[[torch.device], nn.Module]:
    def load(device: torch.device) -> nn.Module:
        model = Model(device=device)
        model.load_state_dict(torch.load(path, map_location=device))
        return model

    return load


def test_inference_time(model: nn.Module, device=torch.device("cpu")):
    test_dataset = CustomDataset(
        train=False, cache=False, data_path=f"{current_path}/../data/datasets"
    )
    test_dataset = torch.stack(test_dataset.images)
    test_dataset = TensorDataset(test_dataset)

    batch_size = 100  # TODO: add the other parameters (device, ...)

    t1 = transform_dataset_with_transforms(test_dataset)
    t2 = transform_dataset_with_model(test_dataset, model, batch_size, device)
    print(
        f"Sequential transforming each image took: {t1} on CPU. \n"
        f"Using a model with batch_size: {batch_size} took {t2} on {device.type}. \n"
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
    model.eval()  # TODO: uncomment this
    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=2, pin_memory=device == "cuda"
    )
    for images in dataloader:
        model(x=images)  # TODO: uncomment this


if __name__ == "__main__":
    freeze_support()
    main()
