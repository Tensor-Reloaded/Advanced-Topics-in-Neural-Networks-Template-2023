import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
from torchvision.transforms import v2

__all__ = ['RandomGaussianBlur', 'RandomRotation', 'Padding']


class Padding:
    def __init__(self):
        self.pad_transform = v2.Pad(padding=10)

    def __call__(self, image):
        return self.pad_transform(image)


class RandomGaussianBlur:
    def __init__(self):
        self.random_blur = transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0))

    def __call__(self, image):
        return self.random_blur(image)


class RandomRotation:
    def __init__(self):
        self.rotation_angle = np.random.randint(0, 360)

    def __call__(self, image):
        return TF.rotate(image, self.rotation_angle)