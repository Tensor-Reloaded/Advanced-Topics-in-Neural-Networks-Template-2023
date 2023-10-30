import os
import random
import re
import datetime
import PIL.Image as PIL_Image
from functools import lru_cache

import torch
import torch.utils.data
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader


class Dataset(torch.utils.data.Dataset):
    def __init__(self, device, dataset_path, transforms=None):
        self.data = []
        self.dataset_path = dataset_path
        self.device = device
        self.transforms = transforms
        self.load_data(self.dataset_path)

    @staticmethod
    def compute_datetime(name):
        pattern = r"global_monthly_(\d{4})_(\d{2})_"
        match = re.search(pattern, name)
        return datetime.datetime(int(match.group(1)), int(match.group(2)), 1)

    @staticmethod
    def compute_time_skip(start_date, end_date):
        return (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)

    def add_data(self, start_image, end_image, time_skip):
        if self.transforms:
            self.data.append((self.transforms(start_image), self.transforms(end_image), time_skip))
        else:
            self.data.append((F.pil_to_tensor(start_image), F.pil_to_tensor(end_image), time_skip))

    def load_data(self, path: str):
        for folder in os.listdir(path):
            location_folder_path = os.path.join(self.dataset_path, folder, 'images')
            if not os.path.isdir(location_folder_path):
                continue

            locations_data = []

            for file in os.listdir(location_folder_path):
                image_path = os.path.join(location_folder_path, file)
                if not os.path.isfile(image_path):
                    continue
                locations_data.append((image_path, Dataset.compute_datetime(file)))

            sorted_data = sorted(locations_data, key=lambda x: x[1])  # sort by date

            for i in range(len(sorted_data) - 1):
                for j in range(i + 1, len(sorted_data)):
                    start_image_path, start_date = sorted_data[i]
                    end_image_path, end_date = sorted_data[j]
                    time_skip = Dataset.compute_time_skip(start_date, end_date)

                    start_image = PIL_Image.open(start_image_path)
                    end_image = PIL_Image.open(end_image_path)
                    self.add_data(start_image, end_image, time_skip)

                    # Rotation augmentation
                    angle = random.uniform(-90, 90)
                    rotated_start_image = start_image.copy().rotate(angle)
                    rotated_end_image = end_image.copy().rotate(angle)
                    self.add_data(rotated_start_image, rotated_end_image, time_skip)

        self.data = self.load_on_device(self.data, self.device)

    @staticmethod
    def load_on_device(data, device):
        for index in range(0, len(data)):
            data[index] = (
                data[index][0].to(device=device, non_blocking=device == "cuda"),
                data[index][1].to(device=device, non_blocking=device == "cuda"),
                data[index][2],
            )

        return data

    def get_data_loaders(self):
        train_size = int(0.7 * len(self.data))
        val_size = int(0.15 * len(self.data))
        test_size = len(self.data) - train_size - val_size
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(self,
                                                                                 [train_size, val_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64)
        test_loader = DataLoader(test_dataset, batch_size=64)

        return train_loader, val_loader, test_loader

    def __len__(self):
        return len(self.data)

    @lru_cache(maxsize=None)
    def __getitem__(self, index):
        return self.data[index]
