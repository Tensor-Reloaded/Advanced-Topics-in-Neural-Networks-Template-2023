from torchvision import datasets
import torch
import typing as t


class MNIST:
    _device: torch.device
    _data: torch.Tensor

    def __init__(
        self,
        path: str = "../data",
        train: bool = False,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self._device = device

        dataset = datasets.MNIST(root=path, train=train, download=True, transform=None)

        vectorised_labels = self._vectorise_labels(dataset.targets)
        vectorised_dataset = self._vectorise_dataset(dataset.data)

        self._data = (vectorised_dataset, vectorised_labels)

    def _vectorise_labels(self, labels: torch.Tensor) -> torch.Tensor:
        unique_labels = torch.unique(labels).tolist()
        data = torch.Tensor(len(labels), len(unique_labels)).to(device=self._device)

        for i in range(0, len(labels)):
            data[i][labels[i]] = 1

        return data.view(data.shape[0], -1)

    def _vectorise_dataset(self, dataset: torch.Tensor) -> torch.Tensor:
        return dataset.view(dataset.shape[0], -1).type(torch.float32)

    def randomise(self) -> t.Self:
        indexes = torch.randperm(self._data[0].shape[0])
        random_data = self._data[0][indexes].to(device=self._device)
        random_labels = self._data[1][indexes].to(device=self._device)

        self._data = (random_data, random_labels)

        return self

    @property
    def data(self) -> t.Tuple[torch.Tensor, torch.Tensor]:
        return self._data
