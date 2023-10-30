import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

class Crop:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        numpy_img = img.numpy()
        numpy_img = (numpy_img * 255).astype(np.uint8)
        pil_img = Image.fromarray(numpy_img)
        img = transforms.Compose([transforms.CenterCrop(self.size), transforms.ToTensor()])(pil_img)
        img = img.permute(1, 2, 0)
        return img

class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        numpy_img = img.numpy()
        numpy_img = (numpy_img * 255).astype(np.uint8)
        pil_img = Image.fromarray(numpy_img)
        img = transforms.Compose([transforms.Resize(self.size), transforms.ToTensor()])(pil_img)
        img = img.permute(1, 2, 0)
        return img

class HorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img1, img2):
        numpy_img1 = img1.numpy()
        numpy_img1 = (numpy_img1 * 255).astype(np.uint8)
        pil_img1 = Image.fromarray(numpy_img1)
        numpy_img2 = img2.numpy()
        numpy_img2 = (numpy_img2 * 255).astype(np.uint8)
        pil_img2 = Image.fromarray(numpy_img2)
        img1 = transforms.Compose([transforms.RandomHorizontalFlip(self.p), transforms.ToTensor()])(pil_img1)
        img2 = transforms.Compose([transforms.RandomHorizontalFlip(self.p), transforms.ToTensor()])(pil_img2)
        img1 = img1.permute(1, 2, 0)
        img2 = img2.permute(1, 2, 0)
        return img1, img2

class RandomRotate:
    def __init__(self, max_angle=30):
        self.max_angle = max_angle

    def __call__(self, img1, img2):
        angle = torch.randint(0, self.max_angle, (1,)).item()
        numpy_img1 = img1.numpy()
        numpy_img1 = (numpy_img1 * 255).astype(np.uint8)
        pil_img1 = Image.fromarray(numpy_img1)
        numpy_img2 = img2.numpy()
        numpy_img2 = (numpy_img2 * 255).astype(np.uint8)
        pil_img2 = Image.fromarray(numpy_img2)
        img1 = transforms.Compose([transforms.RandomRotation(angle), transforms.ToTensor()])(pil_img1)
        img2 = transforms.Compose([transforms.RandomRotation(angle), transforms.ToTensor()])(pil_img2)
        img1 = img1.permute(1, 2, 0)
        img2 = img2.permute(1, 2, 0)
        return img1, img2