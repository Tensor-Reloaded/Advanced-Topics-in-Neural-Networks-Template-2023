from torch import Tensor
from torch.utils.data import Dataset
from typing import Callable
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2


class CIFAR10Dataset(Dataset):
    def __init__(self, is_train: bool, transforms: list[Callable], cached: bool = True):
        self.transforms = transforms
        dataset_obj = self.__get_dataset(is_train, transforms)
        self.dataset: Dataset | tuple = self.__make_cache(dataset_obj) if cached else dataset_obj

    @staticmethod
    def __get_dataset(is_train: bool, transforms: list[Callable]) -> Dataset:
        return CIFAR10(root="./../../data", train=is_train, transform=v2.Compose(transforms), download=True)

    @staticmethod
    def __make_cache(dataset: Dataset) -> tuple:
        return tuple([item for item in dataset])

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.dataset[idx]


