import torch


class Flatten:
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return image.reshape(-1)


class ToFloat:
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return image.to(torch.float32)
