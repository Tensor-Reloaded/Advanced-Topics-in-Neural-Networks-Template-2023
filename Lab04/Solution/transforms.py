import torch
import torchvision.transforms as transforms


class RotateImageWrapper:
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


class ColorJitterWrapper:
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, image):
        image = image.permute(2, 0, 1)
        image = transforms.ToPILImage()(image)
        color_jitter = transforms.ColorJitter(
            brightness=0.2,  # Random brightness adjustment between -0.2 and 0.2
            contrast=0.2,  # Random contrast adjustment between -0.2 and 0.2
            saturation=0.2,  # Random saturation adjustment between -0.2 and 0.2
            hue=0.2  # Random hue adjustment between -0.2 and 0.2
        )
        image = color_jitter(image)
        image = transforms.ToTensor()(image)

        image = image.permute(1, 2, 0)
        return image
