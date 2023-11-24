#!/usr/bin/env python3
import os
import torch
import torch.utils.data as torch_data
import torchvision
from torchvision.transforms import ToTensor, Normalize
from nn.metered_model_trainer import MeteredNeuralNetworkTrainer
from nn.util import get_default_device
from nn.dataset import CachedDataset
from nn.transforms import OneHot
from nn.model import NeuralNetwork
from util.util import Timer


def main():
    device = get_default_device()
    dataset_path = os.path.join(os.path.dirname(__file__), "../data/datasets")
    exports_path = os.path.join(os.path.dirname(__file__), "../data/exports")
    logs_path = os.path.join(os.path.dirname(__file__), "../data/logs")
    transforms = torchvision.transforms.Compose(
        [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
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

    base_model = NeuralNetwork(
        input_size=16,
        output_size=10,
        device=device,
    )
    compiled_model = torch.compile(base_model)
    traced_model = torch.jit.trace(
        base_model,
        next(iter(batched_train_dataset))[0],
    )
    scripted_model = torch.jit.script(base_model)

    metered_base_model = MeteredNeuralNetworkTrainer(
        neural_network=base_model,
        loss_function=torch.nn.CrossEntropyLoss,
        optimiser=torch.optim.Adam,
        learning_rate=0.001,
        device=device,
        log_directory=logs_path,
    )
    metered_compiled_model = MeteredNeuralNetworkTrainer(
        neural_network=compiled_model,
        loss_function=torch.nn.CrossEntropyLoss,
        optimiser=torch.optim.Adam,
        learning_rate=0.001,
        device=device,
        exports_path=exports_path,
        log_directory=logs_path,
    )
    metered_traced_model = MeteredNeuralNetworkTrainer(
        neural_network=traced_model,
        loss_function=torch.nn.CrossEntropyLoss,
        optimiser=torch.optim.Adam,
        learning_rate=0.001,
        device=device,
        exports_path=exports_path,
        log_directory=logs_path,
    )
    metered_scripted_model = MeteredNeuralNetworkTrainer(
        neural_network=scripted_model,
        loss_function=torch.nn.CrossEntropyLoss,
        optimiser=torch.optim.Adam,
        learning_rate=0.001,
        device=device,
        exports_path=exports_path,
        log_directory=logs_path,
    )

    for model in [
        metered_compiled_model,
        metered_traced_model,
        metered_scripted_model,
        metered_base_model,
    ]:
        print(f"Running the model on the device: {device}")

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
        print(f"Run finished in: {timer()}s")


if __name__ == "__main__":
    main()
