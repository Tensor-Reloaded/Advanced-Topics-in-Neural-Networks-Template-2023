import torch
from torchvision import transforms


class ToFloat:
    def __call__(self, image):
        return image.to(torch.float32)


class Transform:
    def __init__(self):
        self.transforms = transforms.Compose([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
            # transforms.RandomErasing(p=0.2, scale=(0.01, 0.01), value=1),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToFloat()
        ])
