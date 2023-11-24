#!/usr/bin/env python3
import os
import torch
import torch.utils.data as torch_data
import torchvision
from torchvision.transforms import ToTensor, Normalize
from nn.util import get_default_device
from nn.dataset import CachedDataset
from nn.transforms import OneHot
from nn.model import NeuralNetwork
from nn.model_trainer import NeuralNetworkTrainer
from util.util import Timer


def main():
    device = get_default_device()
    dataset_path = os.path.join(os.path.dirname(__file__), "../data/datasets")
    exports_path = os.path.join(os.path.dirname(__file__), "../data/exports")
    transforms = torchvision.transforms.Compose(
        [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    target_transforms = OneHot(range(0, 10))

    validation_dataset = torchvision.datasets.CIFAR10(
        root=dataset_path,
        download=True,
        train=False,
        transform=transforms,
        target_transform=target_transforms,
    )
    cached_validation_dataset = CachedDataset(dataset=validation_dataset, cache=True)
    batched_validation_dataset = torch_data.DataLoader(
        dataset=cached_validation_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=2,
        pin_memory=device == "cuda",
    )

    model = NeuralNetwork(
        input_size=16,
        output_size=10,
        device=device,
    )
    model.load_state_dict(torch.load(f"{exports_path}/1700850952457595555.pt", map_location=device))
    model.eval()
    model_trainer = NeuralNetworkTrainer(
        neural_network=model,
        loss_function=torch.nn.CrossEntropyLoss,
        optimiser=torch.optim.Adam,
        learning_rate=0.001,
        device=device,
        exports_path="/tmp"
    )

    timer = Timer()
    results = model_trainer.run_validation(
        batched_validation_dataset=batched_validation_dataset
    )

    print(f"Validation accuracy for the preloaded model: {results[1] * 100:>6.2f}%")
    print(f"Finished in: {timer()}s")


if __name__ == "__main__":
    main()
