from torchvision.transforms import v2

__all__ = ['ImageTransform']

class ImageTransform:
    def __init__(self):
        self.transform_seq = v2.Compose([
        v2.RandomAutocontrast(p=0.4),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(10),
        v2.Pad(padding=5),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
    ])

    def __call__(self, feature):
        return self.transform_seq(feature)

