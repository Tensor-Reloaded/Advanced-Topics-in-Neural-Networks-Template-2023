import os
import re
import random
import datetime
import torch
import PIL.Image as PIL_Image
from functools import lru_cache
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms.functional import pil_to_tensor


class SpacenetDataset(Dataset):
    def __init__(self, dataset_dir_path: str, transform_functions = None, device: str = "cpu"):
        self.device = device
        self.transform_functions = transform_functions
        self.load_dataset(dataset_dir_path)

    @staticmethod
    def extract_date_for_image(filename: str):
        timestamp_regexp = re.compile("global_monthly_(\d{4})_(\d{2})_")
        match = timestamp_regexp.search(filename)
        return datetime.datetime(int(match.group(1)), int(match.group(2)), 1) if match else None

    def load_dataset(self, dir_path: str):
        samples = []
        for location in os.listdir(dir_path):
            images_location_path = os.path.join(dir_path, location, "images")
            if not os.path.isdir(images_location_path):
                raise FileNotFoundError("Dataset directory not found")
            
            location_images = []
            for filename in os.listdir(images_location_path):
                image_full_path = os.path.join(images_location_path, filename)
                image = PIL_Image.open(image_full_path)
                
                image_date = self.extract_date_for_image(filename)
                location_images.append((image, image_date))

            sorted_location_images = sorted(location_images, key = lambda x: x[1])

            for i in range(len(sorted_location_images) - 1):
                for j in range(i+1, len(sorted_location_images)):
                    start_image, start_date = sorted_location_images[i] 
                    end_image, end_date = sorted_location_images[j] 

                    start_image_tensor = pil_to_tensor(start_image)
                    end_image_tensor = pil_to_tensor(end_image)

                    time_skip = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
                    
                    if self.transform_functions:
                        samples.append((self.transform_functions(start_image_tensor), self.transform_functions(end_image_tensor), time_skip))
                    else:
                        samples.append((start_image_tensor, end_image_tensor, time_skip))

                    # Bonus: Random rotation augmentation
                    rotation_angle = random.uniform(0, 360)
                    start_image_rotated_tensor = pil_to_tensor(start_image.copy().rotate(rotation_angle))
                    end_image_rotated_tensor = pil_to_tensor(end_image.copy().rotate(rotation_angle))

                    if self.transform_functions:
                        samples.append((self.transform_functions(start_image_rotated_tensor), self.transform_functions(end_image_rotated_tensor), time_skip))
                    else:
                        samples.append((start_image_rotated_tensor, end_image_rotated_tensor, time_skip))

        self.dataset = self.load_on_device(samples, self.device)
        print("Successfully loaded dataset!")

    def get_data_loaders(self):
        train_dataset, validation_dataset, test_dataset = random_split(self.dataset, [0.7, 0.15, 0.15])
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        validation_loader = DataLoader(validation_dataset, batch_size=32)
        test_loader = DataLoader(test_dataset, batch_size=32)

        return train_loader, validation_loader, test_loader

    def load_on_device(self, dataset, device: str):
        for index in range(0, len(dataset)):
            dataset[index] = (
                dataset[index][0].to(device=device, non_blocking=device == "cuda"),
                dataset[index][1].to(device=device, non_blocking=device == "cuda"),
                dataset[index][2],
            )

        return dataset

    def get_image_shape(self):
        return self.dataset[0][0].shape
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    @lru_cache(maxsize=None)
    def __getitem__(self, index):
        return self.dataset[index]
