import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['RegressionEx3', 'RegressionEx2', 'OneHot', 'WineFeatureGaussianNoise']


class OneHot:
    def __init__(self, classes):
        self.classes = torch.tensor(classes, dtype=torch.long)

    def __call__(self, label):
        return F.one_hot(torch.where(self.classes == label)[0], len(self.classes)).squeeze(0).float()


class RegressionEx2:
    def __init__(self, classes):
        no_labels = len(classes)
        values = torch.zeros((no_labels,))

        length = 1 / (classes.max() - classes.min())
        for index in range(1, no_labels - 1):
            values[index] = values[index - 1] + length
        values[no_labels - 1] = 1

        self.values = values
        self.start = classes.min()

    def __call__(self, label):
        return self.values[label - self.start]


class RegressionEx3:
    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, label):
        return torch.tensor(0.0) if label < self.threshold else torch.tensor(1.0)


class WineFeatureGaussianNoise:
    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std

    def __call__(self, feature):
        return feature + torch.randn_like(feature) * self.std + self.mean
