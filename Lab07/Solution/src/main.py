#!/usr/bin/env python3
import torch
import torch.utils.data as torch_data
import torchvision
from torchvision.transforms import v2
from torchvision.transforms import Normalize, RandomResizedCrop, RandomHorizontalFlip
from nn.metered_trainable_model import MeteredTrainableNeuralNetwork
from nn.util import get_default_device
from nn.dataset import CachedDataset
from nn.transforms import OneHot


def main():
    device = get_default_device()
    dataset_path = "../data/datasets"
    logs_path = "../data/logs"
    transforms = torchvision.transforms.Compose(
        [
            v2.ToImageTensor(),
            v2.ToDtype(torch.float32),
            RandomResizedCrop(32),
            RandomHorizontalFlip(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    target_transforms = OneHot(range(0, 10))

    training_dataset = torchvision.datasets.CIFAR10(
        root=dataset_path,
        download=True,
        train=True,
        transform=transforms,
        target_transform=target_transforms,
    )
    validation_dataset = torchvision.datasets.CIFAR10(
        root=dataset_path,
        download=True,
        train=False,
        transform=transforms,
        target_transform=target_transforms,
    )

    cached_training_dataset = CachedDataset(dataset=training_dataset, cache=True)
    cached_validation_dataset = CachedDataset(dataset=validation_dataset, cache=True)

    batched_train_dataset = torch_data.DataLoader(
        dataset=cached_training_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=2,
        pin_memory=device == "cuda",
    )
    batched_validation_dataset = torch_data.DataLoader(
        dataset=cached_validation_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=2,
        pin_memory=device == "cuda",
    )

    model = MeteredTrainableNeuralNetwork(
        input_size=32 * 32,
        output_size=10,
        loss_function=torch.nn.CrossEntropyLoss,
        optimiser=torch.optim.Adam,
        learning_rate=0.001,
        device=device,
        log_directory=logs_path,
    )

    timer = Timer()
    before_training_results = model.run_validation(
        batched_validation_dataset=batched_validation_dataset
    )
    model.run(
        batched_training_dataset=batched_train_dataset,
        batched_validation_dataset=batched_validation_dataset,
        epochs=25,
    )
    after_training_results = model.run_validation(
        batched_validation_dataset=batched_validation_dataset
    )

    print(
        f"Validation accuracy before training: {before_training_results[1] * 100:>6.2f}%"
    )
    print(
        f"Validation accuracy after  training: {after_training_results[1] * 100:>6.2f}%"
    )
    print(
        f"Run finished in: {timer()}s"
    )

if __name__ == "__main__":
    main()
