import torchvision.transforms as transforms
from torchvision.transforms import functional as F


class Crop:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        return F.resized_crop(img, 0, 0, self.size, self.size, self.size)


class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        return F.resize(img, self.size)


class HorizontalFlip:
    def __init__(self):
        pass

    def __call__(self, img):
        return F.hflip(img)


class Rotate:
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, img):
        return F.rotate(img, self.angle)
