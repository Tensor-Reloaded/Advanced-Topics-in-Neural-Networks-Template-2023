import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['OneHot', 'WineFeatureGaussianNoise']

class OneHot:
    def __init__(self, classes):
        self.classes = torch.tensor(classes, dtype=torch.long)

    def __call__(self, label):
        return F.one_hot(torch.where(self.classes == label)[0], len(self.classes)).squeeze(0).float()
    
class WineFeatureGaussianNoise:
    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std

    def __call__(self, feature):
        return feature + torch.randn_like(feature) * self.std + self.mean