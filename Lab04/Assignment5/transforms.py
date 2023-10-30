from typing import Union, List, Tuple

import cv2
import torch
import torchvision
from torch import Tensor
from torchvision.transforms import v2


class MinMaxNormalization:
    def __init__(self, instances: Union[str, List[int]] = 'all', minmax: bool = False,
                 minim=0, maxim=255):
        self.instances = instances
        self.maxim = maxim
        self.minim = minim
        self.minmax = minmax

    def __call__(self, sample_par: Union[Tensor, List[Tensor]]):
        if type(sample_par) == list:
            sample = list(sample_par)
        else:
            sample = Tensor(sample_par)
        if type(sample) == list:
            if self.instances == 'all':
                self.instances = range(len(sample))
            if self.minmax:
                self.minim = sample[self.instances].min()
                self.maxim = sample[self.instances].max()
            for instance in self.instances:
                sample[instance] -= self.minim
                sample[instance] /= (self.maxim - self.minim)
        else:
            if self.minmax:
                self.minim = sample.min()
                self.maxim = sample.max()
            sample -= self.minim
            sample /= (self.maxim - self.minim)
        return sample


class StandardNormalization:
    def __init__(self, instances: Union[str, List[int]] = 'all', mean=0, std=1):
        self.instances = instances
        self.mean = mean
        self.std = std

    def __call__(self, sample_par: Union[Tensor, List[Tensor]]):
        if type(sample_par) == list:
            sample = list(sample_par)
        else:
            sample = Tensor(sample_par)
        if type(sample) == list:
            if self.instances == 'all':
                self.instances = range(len(sample))
            for instance in self.instances:
                sample[instance] -= torchvision.transforms.Normalize(
                    mean=self.mean, std=self.std)(sample[instance])
        else:
            sample = torchvision.transforms.Normalize(
                mean=self.mean, std=self.std)(sample)
        return sample


class ChangeType:
    def __init__(self, dtype, instances: Union[str, List[int]] = 'all'):
        self.instances = instances
        self.desired_type = dtype

    def __call__(self, sample_par: Union[Tensor, List[Tensor]]):
        if type(sample_par) == list:
            sample = list(sample_par)
        else:
            sample = Tensor(sample_par)
        if type(sample) == list:
            if self.instances == 'all':
                self.instances = range(len(sample))
            for instance in self.instances:
                sample[instance] = v2.ToDtype(dtype=self.desired_type, scale=True)(sample[instance])
        else:
            sample = v2.ToDtype(dtype=self.desired_type, scale=True)(sample)
        return sample


class Crop:
    def __init__(self, instances: Union[str, List[int]] = 'all'):
        self.instances = instances

    def __call__(self, sample_par: Union[Tensor, List[Tensor]]):
        if type(sample_par) == list:
            sample = list(sample_par)
        else:
            sample = Tensor(sample_par)
        if type(sample) == list:
            if self.instances == 'all':
                self.instances = range(len(sample))
            for instance in self.instances:
                sample[instance] = v2.RandomResizedCrop(sample[instance].shape)(sample[self.instances])
        else:
            sample = v2.RandomResizedCrop(sample.shape)(sample)
        return sample


class RandomRotation:
    def __init__(self, instances: Union[str, List[int]] = 'all'):
        self.instances = instances

    def __call__(self, sample_par: Union[Tensor, List[Tensor]]):
        if type(sample_par) == list:
            sample = list(sample_par)
        else:
            sample = Tensor(sample_par)
        if type(sample) == list:
            if self.instances == 'all':
                self.instances = range(len(sample))
            for instance in self.instances:
                sample[instance] = v2.RandomRotation(degrees=10)(sample[instance])
        else:
            sample = v2.RandomRotation(sample.shape)(sample)
        return sample


class ColorChange:
    def __init__(self, instances: Union[str, List[int]] = 'all'):
        self.instances = instances

    def __call__(self, sample_par: Union[Tensor, List[Tensor]]):
        if type(sample_par) == list:
            sample = list(sample_par)
        else:
            sample = Tensor(sample_par)
        if type(sample) == list:
            if self.instances == 'all':
                self.instances = range(len(sample))
            for instance in self.instances:
                sample[instance] = v2.ColorJitter(brightness=0.5, contrast=1,
                                                  saturation=0.1, hue=0.5)(sample[instance])
        else:
            sample = v2.ColorJitter(brightness=0.5, contrast=1,
                                    saturation=0.1, hue=0.5)(sample)
        return sample


class ImageToTensor:
    def __init__(self, instances: Union[str, List[int]] = 'all'):
        self.instances = instances

    def __call__(self, sample_par: Union[str, List[str]]):
        if type(sample_par) == list:
            sample = list(sample_par)
        else:
            sample = Tensor(sample_par)
        if type(sample) == list:
            if self.instances == 'all':
                self.instances = range(len(sample))
            for instance in self.instances:
                sample[instance] = torch.from_numpy(cv2.imread(sample[instance], cv2.IMREAD_COLOR))
        else:
            sample = torch.from_numpy(cv2.imread(sample, cv2.IMREAD_COLOR))
        return sample


class NumberToTensor:
    def __init__(self, instances: Union[str, List[int]] = 'all'):
        self.instances = instances

    def __call__(self, sample_par: Union[int, List[int]]):
        if type(sample_par) == list:
            sample = list(sample_par)
        else:
            sample = Tensor(sample_par)
        if type(sample) == list:
            if self.instances == 'all':
                self.instances = range(len(sample))
            for instance in self.instances:
                sample[instance] = Tensor([sample[instance]])
        else:
            sample = Tensor(sample)
        return sample


class FeatureLabelsSplit:
    def __init__(self, features: List[int], labels: List[int]):
        self.features = features
        self.labels = labels

    def __call__(self, sample):
        features_tensor = Tensor(sample[self.features[0]])
        labels_tensor = Tensor(sample[self.labels[0]])
        for instance in self.features[1:]:
            features_tensor = torch.cat((features_tensor, sample[instance]))
        for instance in self.labels[1:]:
            labels_tensor = torch.stack((labels_tensor, sample[instance]))
        return [features_tensor, labels_tensor]


class GroupTensors:
    def __init__(self, instances: Union[str, List[int]] = 'all', position: int = 0):
        self.instances = instances
        self.position = position

    def __call__(self, sample_par):
        if type(sample_par) == list:
            sample = list(sample_par)
        else:
            sample = Tensor(sample_par)
        if self.instances == 'all':
            self.instances = range(len(sample))
        result = Tensor(sample[self.instances[0]])
        for instance in self.instances[1:]:
            result = torch.stack((result, sample[instance]))
        deleted = 0
        for instance in self.instances:
            del sample[instance - deleted]
        sample.insert(self.position, result)
        return sample


class ReshapeTensors:
    def __init__(self, shape: Union[Tuple, int],
                 instances: Union[str, List[int]] = 'all'):
        self.instances = instances
        self.shape = shape

    def __call__(self, sample_par):
        if type(sample_par) == list:
            sample = list(sample_par)
        else:
            sample = Tensor(sample_par)
        if type(sample) == list:
            if self.instances == 'all':
                self.instances = range(len(sample))
            for instance in self.instances:
                sample[instance] = sample[instance].reshape(self.shape)
        else:
            sample = sample.reshape(self.shape)
        return sample


class UngroupTensors:
    def __init__(self, instance: int):
        self.instance = instance
        self.position = 0

    def __call__(self, sample_par):
        if type(sample_par) == list:
            sample = list(sample_par)
        else:
            sample = Tensor(sample_par)
        result = list(sample[self.instance].split(1))
        sample = sample[:self.instance] + result + sample[(self.instance + 1):]
        return sample
