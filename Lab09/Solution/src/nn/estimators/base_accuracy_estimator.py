import torch


class BaseAccuracyEstimator:
    def update(self, y_hat: torch.Tensor, y: torch.Tensor) -> None:
        pass

    def get(self) -> float:
        return 0.0
