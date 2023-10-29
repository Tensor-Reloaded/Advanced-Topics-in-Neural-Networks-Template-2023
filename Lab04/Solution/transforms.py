import random
import torch
import torchvision


class RandomRotation:
    __angle: float

    def __init__(self, min_angle: float, max_angle: float):
        self.__angle = random.uniform(min_angle, max_angle)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return torchvision.transforms.functional.rotate(image, self.__angle)

class Flatten:
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return image.reshape(-1)

class ToFloat:
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return image.to(torch.float32)
