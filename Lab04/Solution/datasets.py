import os
from functools import lru_cache

import cv2 as cv
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

__all__ = ['ImageDataset']

from datetime import datetime


# Custom dataset class
class ImageDataset(Dataset):

    def __init__(self, dataset_file, transform_list=None, device="cpu"):
        self.transform_list = transform_list
        self.tuples = []

        location_dirs = os.listdir(os.path.join(dataset_file))
        for dir in location_dirs:  # L15-format
            files = [os.path.join(dataset_file, dir, "images", file)
                     for file in os.listdir(os.path.join(dataset_file, dir, "images"))]

            for start_file in files:
                for end_file in files:  # consider they might not be ordered for some reason
                    start_date_str = start_file.split("global_monthly_")[1].split("_mosaic")[0]
                    start_date = datetime.strptime(start_date_str, "%Y_%m")

                    end_date_str = end_file.split("global_monthly_")[1].split("_mosaic")[0]
                    end_date = datetime.strptime(end_date_str, "%Y_%m")
                    if end_date < start_date:
                        start_date, end_date = end_date, start_date

                    self.tuples.append((start_file, end_file, ImageDataset.months(end_date, start_date)))

    @staticmethod
    def months(d1, d2):
        return d1.month - d2.month + 12 * (d1.year - d2.year)

    def __len__(self):
        return len(self.tuples)

    @lru_cache(maxsize=10_000)
    def __getitem__(self, idx):
        input_file, output_file, delta_time = self.tuples[idx]
        input_img = torch.from_numpy(cv.imread(input_file))
        output_img = torch.from_numpy(cv.imread(output_file))

        for transform in self.transform_list:
            input_img = transform(input_img)
            output_img = transform(input_img)
        return (input_img / 255.0).flatten(), (output_img / 255.0).flatten(), delta_time
