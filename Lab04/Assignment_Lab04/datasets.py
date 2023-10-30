import torch
from torch.utils.data import Dataset
import os
from PIL import Image
from typing import Callable
import random
from torchvision.transforms import ToTensor


def assure_same_transformation(transformation_type: Callable) -> Callable:
    def transform_pair_of_images(start_image, end_image):
        random_state = torch.get_rng_state()
        start_image = transformation_type(start_image)
        torch.set_rng_state(random_state)
        end_image = transformation_type(end_image)
        return start_image, end_image
    return transform_pair_of_images


# Custom dataset class
class BuildingsDataset(Dataset):
    def __init__(self, folder_path, feature_transforms=None, label_transforms=None):
        self.feature_transforms = feature_transforms if feature_transforms is not None else []
        self.label_transforms = label_transforms if label_transforms is not None else []

        list_of_tuples = []  # This will be the returned dataset used in all operations performed within nn

        dataset_path = ''
        dataset_content = []
        list_of_inputs = []

        for files in os.listdir(folder_path):
            if "Dataset" in files:
                dataset_path = str(folder_path) + "\\" + files

        for file in os.listdir(dataset_path):
            dataset_content.append(dataset_path + "\\" + file)

        for folder in dataset_content:
            paths_of_input_elements = []
            for file in os.listdir(folder):
                item_path = folder + "\\" + file
                for item in os.listdir(item_path):
                    image_path = item_path + "\\" + item
                    paths_of_input_elements.append(image_path)
            list_of_inputs.append(paths_of_input_elements)

        for location in list_of_inputs:
            for index in range(len(location) - 1):
                # I should return a tuple with the following form (start_image, end_image, time_skip)
                start = location[index]  # This is the path of an image (start_image from tuple)
                start_image = Image.open(start)
                start_image_torch = ToTensor()(start_image)

                end = location[index + 1]  # This is the path of an image (end_image from tuple)
                end_image = Image.open(end)
                end_image_torch = ToTensor()(end_image)

                # Here I determine the last parameter of the tuple (time_skip)
                date_start_s = int(start.rindex("monthly") + len("monthly_"))
                date_start_e = int(start.rindex("_mosaic"))
                date_start = start[date_start_s:date_start_e]

                date_end_s = int(end.rindex("monthly") + len("monthly_"))
                date_end_e = int(end.rindex("_mosaic"))
                date_end = end[date_end_s:date_end_e]

                year_start = int(date_start[0:4])
                year_end = int(date_end[0:4])

                month_start = int(date_start[len(date_start) - 2:])
                month_end = int(date_end[len(date_end) - 2:])

                time_skip = (12 * year_end + month_end) - (12 * year_start + month_start)

                result_tuple = (start_image_torch, end_image_torch, time_skip)
                list_of_tuples.append(result_tuple)

                if self.feature_transforms:
                    for t in self.feature_transforms:
                        same_transformer = assure_same_transformation(t)
                        transformed_images = same_transformer(start_image, end_image)  # Dataset augmentation -
                        # Rotation, Horizontal/Vertical Flip, Gaussian Blur, Elastic Transform

                        transformed_start_image = ToTensor()(transformed_images[0])
                        transformed_end_image = ToTensor()(transformed_images[1])

                        transformed_result_tuple = (transformed_start_image, transformed_end_image, time_skip)
                        list_of_tuples.append(transformed_result_tuple)

        random.shuffle(list_of_tuples)
        self.input_elements = list_of_tuples

    def __len__(self):
        return len(self.input_elements)

    def __getitem__(self, idx):
        return self.input_elements[idx][0], self.input_elements[idx][1], self.input_elements[idx][2]
