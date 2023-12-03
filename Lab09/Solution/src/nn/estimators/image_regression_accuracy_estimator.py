import torch
from nn.estimators.base_accuracy_estimator import BaseAccuracyEstimator


class ImageRegressionAccuracyEstimator(BaseAccuracyEstimator):
    __precision: float
    __entries: int
    __accuracy: float

    def __init__(self, precision: float = 0.0) -> None:
        super().__init__()

        self.__precision = precision
        self.__entries = 0
        self.__accuracy = 1.0

    def update(self, y_hat: torch.Tensor, y: torch.Tensor) -> None:
        lower_bounds = y - self.__precision
        upper_bounds = y + self.__precision

        is_within_lower_bounds = y_hat >= lower_bounds
        is_within_upper_bounds = y_hat <= upper_bounds

        accurate_pixels = (is_within_lower_bounds & is_within_upper_bounds).sum().item()
        total_pixels = y.numel()

        new_accuracy = accurate_pixels / total_pixels

        self.__accuracy = (self.__entries * self.__accuracy + new_accuracy) / (
            self.__entries + 1
        )
        self.__entries += 1

    def get(self) -> float:
        return self.__accuracy
