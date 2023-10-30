import torch
from torch.utils.data import Dataset
import pandas as pd
import cv2
import matplotlib.pyplot as plt

__all__ = ['ImageDataset']


# Custom dataset class
class ImageDataset(Dataset):
    def __init__(self, dataset_file, split_indices=None, feature_transforms=None, label_transforms=None, combined_random_transforms=None):
        self.feature_transforms = feature_transforms if feature_transforms is not None else []
        self.label_transforms = label_transforms if feature_transforms is not None else []
        self.combined_random_transforms = combined_random_transforms if combined_random_transforms is not None else []

        df = pd.read_csv(dataset_file)
        if split_indices is None:
            split_indices = torch.arange(len(df))
        df = df.iloc[split_indices]
        self.features = df.drop('end_image', axis=1).values
        self.labels = df['end_image'].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        features = self.features[idx][0]  # Don't transform the original features
        time_skip = int(self.features[idx][1])
        labels = self.labels[idx]  # Don't transform the original labels
        features = torch.from_numpy(cv2.imread(features, cv2.IMREAD_UNCHANGED))
        labels = torch.from_numpy(cv2.imread(labels, cv2.IMREAD_UNCHANGED))

        for transform in self.feature_transforms:
            features = transform(features)
        for transform in self.label_transforms:
            labels = transform(labels)
        for transform in self.combined_random_transforms:
            features, labels = transform(features, labels)

        features = features.reshape(-1)
        labels = labels.reshape(-1)

        # Normalize features
        features = torch.tensor(features, dtype=torch.float32) / 255
        labels = torch.tensor(labels, dtype=torch.float32) / 255
        return features, time_skip, labels
