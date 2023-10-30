import torch
from torch.utils.data import Dataset

__all__ = ['GlobalImageDataset']

from data_reader import DataReader


# Custom dataset class
class GlobalImageDataset(Dataset):
    def __init__(self, dataset_file, data_transforms=None):
        self.data_transforms = data_transforms if data_transforms is not None else []
        self.time_skip = 6
        data_reader = DataReader(dataset_file, self.time_skip)
        data_reader.read_and_convert_images()
        data_reader.create_dataset()

        self.dataset = data_reader.dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        input_data = self.dataset[idx][0]  # Don't transform the original features
        output_data = self.dataset[idx][1]  # Don't transform the original labels
        for transform in self.data_transforms:
            input_data = transform(input_data)
            output_data = transform(output_data)
        return torch.flatten(input_data), torch.flatten(output_data), self.time_skip
