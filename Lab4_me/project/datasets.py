import os
import re
import itertools
import random
import numpy as np

import cv2
from torch.utils.data import Dataset
import torch

__all__ = ['AgedCityImagesDataset']




# Custom dataset class
class AgedCityImagesDataset(Dataset):
    def __init__(self, data_folder, transform=None):
        self.data_folder = data_folder
        self.transform = transform

        self.city_subfolders = []
        for subfolder in os.listdir(data_folder):
            if os.path.isdir(os.path.join(data_folder, subfolder)):
                self.city_subfolders.append(subfolder)

        self.data = []

        # Read and organize data
        for city_subfolder in self.city_subfolders:
            city_path = os.path.join(data_folder, city_subfolder)

            # List image files in the city folder
            image_files = []
            for file in os.listdir(city_path + '\\images'):
                if file.lower().endswith('.tif'):
                    image_files.append(file)
            image_files = sorted(image_files)


            # global_monthly_2018_02_mosaic_L15-0632E-0892N_2528_4620_13.tif
            dates = []
            for filename in image_files:
                year, month = map(int, re.search(r'(\d{4})_(\d{2})', filename).groups())  #it's important to only take the first group
                dates.append((year, month))

            date_pairs = list(itertools.combinations(enumerate(dates), 2))

            # Calculate time intervals between images
            time_intervals = [self.calculate_time_interval(dates[i], dates[i + 1]) for i in range(len(dates) - 1)]

            flattened_date_pairs = [
                (self.calculate_time_interval(date1, date2), i, j) for (i, date1), (j, date2) in date_pairs
            ]

            date_image_points = []
            for time_difference, i, j in flattened_date_pairs:
                path1 = os.path.join(city_path + '\\images', image_files[i])
                start_image = cv2.imread(path1)

                path2 =os.path.join(city_path + '\\images', image_files[j])
                end_image = cv2.imread(path2)  # Aging based on time difference

                date_skip = time_difference  # Number of months skipped

                date_image_points.append((start_image, end_image, date_skip))

            self.data.extend(date_image_points)


    def calculate_time_interval(self, date1, date2):
        year1, month1 = date1
        year2, month2 = date2

        return (year2 - year1) * 12 + (month2 - month1)
    # Function for data augmentation
    # Function for data augmentation
    def augment_images(image1, image2):
        angle = random.uniform(-10, 10)
        rows1, cols1, _ = image1.shape
        rows2, cols2, _ = image2.shape

        # Apply rotation to both images
        M1 = cv2.getRotationMatrix2D((cols1 / 2, rows1 / 2), angle, 1)
        augmented_image1 = cv2.warpAffine(image1, M1, (cols1, rows1))
        M2 = cv2.getRotationMatrix2D((cols2 / 2, rows2 / 2), angle, 1)
        augmented_image2 = cv2.warpAffine(image2, M2, (cols2, rows2))

        augmented_image1 = cv2.resize(augmented_image1, (128, 128))
        augmented_image2 = cv2.resize(augmented_image2, (128, 128))

        return augmented_image1, augmented_image2

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            start_image, end_image, time_skip = self.data[idx]
        else:
            raise TypeError("Unsupported index type")

        # Normalize pixel values to the range [0, 1]
        image1_normalized = start_image.astype(np.float32) / 255.0
        image2_normalized = end_image.astype(np.float32) / 255.0

        # Convert your NumPy arrays to PyTorch tensors
        image1_tensor = torch.from_numpy(image1_normalized)
        image2_tensor = torch.from_numpy(image2_normalized)

        return image1_tensor, image2_tensor, time_skip