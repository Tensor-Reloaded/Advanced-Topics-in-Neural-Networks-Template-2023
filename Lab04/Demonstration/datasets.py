import torch
from torch.utils.data import Dataset
import pandas as pd

__all__ = ['WineDataset']

# Custom dataset class
class WineDataset(Dataset):
    def __init__(self, dataset_file, split_indices=None, feature_transforms=None, label_transforms=None):
        self.feature_transforms = feature_transforms if feature_transforms is not None else []
        self.label_transforms = label_transforms if feature_transforms is not None else []
        
        df = pd.read_csv(dataset_file)
        if split_indices is None:
            split_indices = torch.arange(len(df))
        df = df.iloc[split_indices]
        self.features = df.drop('quality', axis=1).values
        self.labels = df['quality'].values

        # Normalize features
        self.features = torch.tensor(self.features, dtype=torch.float32)
        self.features = (self.features - self.features.mean(dim=0)) / self.features.std(dim=0)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        features = self.features[idx] # Don't transform the original features
        labels = self.labels[idx] # Don't transform the original labels
        for transform in self.feature_transforms:
            features = transform(features)
        for transform in self.label_transforms:
            labels = transform(labels)
        return features, labels
