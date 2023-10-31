import torch
import torchvision.transforms as transforms
from torchvision.transforms.functional import hflip


class RotateImage:
    def __init__(self, min_angle=-45, max_angle=45):
        self.min_angle = min_angle
        self.max_angle = max_angle

    def __call__(self, image):
        image = image.permute(2, 0, 1)
        image_pil = transforms.ToPILImage()(image)
        rotation_angle = torch.empty(1).uniform_(self.min_angle, self.max_angle)
        rotated_image = transforms.functional.rotate(image_pil, angle=rotation_angle.item())
        rotated_image_tensor = transforms.ToTensor()(rotated_image)

        image = rotated_image_tensor.permute(1, 2, 0)
        return image


class CropImage:
    def __init__(self):
        pass

    def __call__(self, image):
        image = image.permute(2, 0, 1)
        image = transforms.ToPILImage()(image)
        composed_transform = transforms.Compose([
            # Crop a region of interest from the image. The arguments are (top, left, height, width).
            transforms.CenterCrop((64, 64)),
            # Resize the cropped image to 128x128 pixels
            transforms.Resize((128, 128))
        ])
        image = composed_transform(image)
        image = transforms.ToTensor()(image)

        image = image.permute(1, 2, 0)
        return image


class BlurImage:
    def __init__(self):
        self.random_blur = transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0))

    def __call__(self, image):
        return self.random_blur(image)
