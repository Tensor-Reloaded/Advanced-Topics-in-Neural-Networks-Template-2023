from torchvision import datasets
import torch
import typing as t


class MNIST:
    _device: torch.device
    _training_data: torch.Tensor
    _testing_data: torch.Tensor

    def __init__(
        self,
        path: str = "../data",
        train_data_percentage: float = 0.75,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self._device = device

        dataset = datasets.MNIST(root=path, train=False, download=True, transform=None)
        training_data_count = int(dataset.data.shape[0] * train_data_percentage)

        vectorised_labels = self._vectorise_labels(dataset.targets)
        vectorised_dataset = self._vectorise_dataset(dataset.data)
        normalised_dataset = self._normalise(vectorised_dataset)
        training_data, testing_data = self._partition(
            normalised_dataset, vectorised_labels, training_data_count
        )

        self._training_data = training_data
        self._testing_data = testing_data

    def _vectorise_labels(self, labels: torch.Tensor) -> torch.Tensor:
        unique_labels = torch.unique(labels).tolist()
        data = torch.Tensor(len(labels), len(unique_labels)).to(device=self._device)

        for i in range(0, len(labels)):
            data[i][labels[i]] = 1

        return data.view(data.shape[0], -1)

    def _vectorise_dataset(self, dataset: torch.Tensor) -> torch.Tensor:
        return dataset.view(dataset.shape[0], -1).type(torch.float32)

    def _normalise(self, dataset: torch.Tensor) -> torch.Tensor:
        return dataset / torch.max(dataset)

    def _partition(
        self, dataset: torch.Tensor, labels: torch.Tensor, training_data_count: int
    ) -> t.Tuple[
        t.Tuple[torch.Tensor, torch.Tensor, int],
        t.Tuple[torch.Tensor, torch.Tensor, int],
    ]:
        training_data = [
            dataset[:training_data_count].to(device=self._device),
            labels[:training_data_count].to(device=self._device),
            training_data_count,
        ]
        testing_data = [
            dataset[training_data_count:].to(device=self._device),
            labels[training_data_count:].to(device=self._device),
            dataset.shape[0] - training_data_count,
        ]

        return training_data, testing_data

    def randomise_training_data(self) -> t.Self:
        indexes = torch.randperm(self._training_data[0].shape[0])
        random_data = self._training_data[0][indexes].to(device=self._device)
        random_labels = self._training_data[1][indexes].to(device=self._device)

        self._training_data = (random_data, random_labels, self._training_data[2])

        return self

    def randomise_testing_data(self) -> t.Self:
        indexes = torch.randperm(self._testing_data[0].shape[0])
        random_data = self._testing_data[0][indexes].to(device=self._device)
        random_labels = self._testing_data[1][indexes].to(device=self._device)

        self._training_data = (random_data, random_labels, self._testing_data[2])

        return self

    @property
    def training_data(self) -> t.Tuple[torch.Tensor, torch.Tensor, int]:
        return self._training_data

    @property
    def testing_data(self) -> t.Tuple[torch.Tensor, torch.Tensor, int]:
        return self._testing_data
