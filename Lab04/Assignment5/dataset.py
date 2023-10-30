import itertools
import os

from Assignment5 import utils


class Dataset:
    def __init__(self, dataset_path, build_dataset, transformations=None):
        self.transformations = transformations if transformations is not None else []
        self.photo_pairs, self.dataset_size = build_dataset(dataset_path)
        # photo pairs can be any pair input-output, not necessarily consisting in photos

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        photo_pair = self.photo_pairs[idx]  # Don't transform the original features
        for transform in self.transformations:
            photo_pair = transform(photo_pair)
        return photo_pair
