from math import ceil

import torch
from torch.utils.data import Dataset
from typing import Callable, Optional
import os
import datetime
import pandas as pd
from torch import Tensor
from skimage import io


class ImageDataset(Dataset):

    def __init__(self, dataset_folder: str, csv_file_path: str, device, image_transforms: Optional[Callable] = None,
                 load_from_dataset: bool = False):
        self.image_transforms = image_transforms if image_transforms is not None else []
        self.dataset_folder = dataset_folder
        self.csv_file_path = csv_file_path

        if load_from_dataset or not os.path.exists(csv_file_path):
            df = write_dataset_to_csv(dataset_folder, csv_file_path)
        else:
            df = pd.read_csv(csv_file_path)
        self.start_images: [str] = df['start_image'].values
        self.end_images: [str] = df['end_image'].values
        self.month_diffs: [int] = df['months_apart']

    def __len__(self):
        return len(self.start_images)

    def __getitem__(self, index) -> [Tensor, Tensor, Tensor]:
        start_image_name = self.start_images[index]
        end_image_name = self.end_images[index]
        #global_monthly_2018_01_mosaic_L15-0331E-1257N_1327_3160_13.tif
        image_folder = start_image_name.split('_mosaic_')[1].split('.')[0]
        image_folder = os.path.join(self.dataset_folder, image_folder, 'images')

        start_image = torch.tensor(io.imread(os.path.join(image_folder, start_image_name)))
        end_image = torch.tensor(io.imread(os.path.join(image_folder, end_image_name)))
        months_apart = self.month_diffs[index]

        for transform in self.image_transforms:
            state = torch.get_rng_state()
            start_image = transform(start_image)
            torch.set_rng_state(state)
            end_image = transform(end_image)

        return start_image, months_apart, end_image


# Example name: global_monthly_2018_01_mosaic_L15-0331E-1257N_1327_3160_13.tif
def get_nr_months_from_file_name(file_name: str) -> datetime:
    split_name = file_name.split("_", 5)
    year = int(split_name[2])
    month = int(split_name[3])
    return year * 12 + month


def months_diff(start, end):
    return ceil((end - start).days / 30)


def write_dataset_to_csv(dataset_folder: str, csv_file_path: str):
    locations = os.listdir(dataset_folder)
    start_image = []
    end_image = []
    months_apart = []
    for loc in locations:
        current_location_imgs = os.listdir(os.path.join(dataset_folder, loc, "images"))
        for i in range(len(current_location_imgs)):
            for j in range(i + 1, len(current_location_imgs)):
                start_image.append(current_location_imgs[i])
                end_image.append(current_location_imgs[j])
                months_apart.append(
                    get_nr_months_from_file_name(end_image[-1]) - get_nr_months_from_file_name(start_image[-1])
                )
    df = pd.DataFrame({'start_image': start_image, 'end_image': end_image, 'months_apart': months_apart})
    df.to_csv(csv_file_path)

    return df
