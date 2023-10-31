import os
import cv2 as cv
import torch
from torch.utils.data import Dataset
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

    # @staticmethod
    def months(d1, d2):
        return d1.month - d2.month + 12 * (d1.year - d2.year)

    def __len__(self):
        return (len(self.tuples)-2)

    # @lru_cache(maxsize=10_000)
    def __getitem__(self, idx):
        input_file, output_file, delta_time = self.tuples[idx]
        print("int = " + input_file)
        print("out = " + output_file)

        # Testing image loading
        # test_input_file = ".././Homework Dataset/L15-1716E-1211N_6864_3345_13/images"
        # test_output_file = ".././Homework Dataset/L15-1716E-1211N_6864_3345_13/images"
        # input_img = cv.imread(test_input_file)
        # output_img = cv.imread(test_output_file)
        # if input_img is None or output_img is None:
        #     raise Exception(f"Failed to load image from exemple {test_input_file} or {test_output_file}")

        input_img = cv.imread(input_file)
        output_img = cv.imread(output_file)

        if input_img is None or output_img is None:
            # Handle the case where image loading failed
            raise Exception(f"Failed to load image from {input_file} or {output_file}")

        input_img = torch.from_numpy(input_img)
        output_img = torch.from_numpy(output_img)

        for transform in self.transform_list:
            input_img = transform(input_img)
            output_img = transform(output_img)

        return (input_img / 255.0).flatten(), (output_img / 255.0).flatten(), delta_time
