from abc import abstractmethod
import torch


class BaseAccuracyEstimator:
    @abstractmethod
    def update(self, y_hat: torch.Tensor, y: torch.Tensor) -> None:
        pass

    @abstractmethod
    def get(self) -> float:
        return 0.0
