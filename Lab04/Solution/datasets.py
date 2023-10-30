import numpy as np
import torch
from torch.utils.data import Dataset
import os
import cv2 as cv2
import torchvision.transforms.v2 as transforms

__all__ = ['MosaicDataset']


def extract_year_month(text: str) -> tuple[int, int]:
    tokens = text.split('_', 4)
    return int(tokens[2]), int(tokens[3])


# For each image ,we will associate all the images that have a positive time skip between them,
# saved as paths
class MosaicDataset(Dataset):
    def __init__(self, dataset_folder: str, use_random_rotation: bool = True, transformers=None):
        self.use_random_rotation = use_random_rotation
        self.transformers = transformers if transformers is not None else []

        self.features = []
        self.labels = []
        for (directory_path, directory_names, file_names) in os.walk(dataset_folder):
            if len(directory_names) == 0:
                no_files = len(file_names)
                for index1 in range(no_files - 1):
                    for index2 in range(index1 + 1, no_files):
                        year1, month1 = extract_year_month(file_names[index1])
                        year2, month2 = extract_year_month(file_names[index2])
                        if year2 < year1 or (year2 == year1 and month2 < month1):
                            print("Something is wrong in the dataset file names on years and months")
                            exit(0)

                        months_between = 0
                        if year1 == year2:
                            months_between = month2 - month1
                        else:
                            months_between = (year2 - year1 - 1) * 12 + 12 - month1 + month2

                        if months_between <= 0:
                            print(
                                "Something wrong in the dataset file names since we got a negative difference in months")
                            exit(0)

                        self.features.append((directory_path + "\\" + file_names[index1], months_between))
                        self.labels.append(directory_path + "\\" + file_names[index2])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        path_image1, months_between = self.features[idx]
        path_image2 = self.labels[idx]

        image1 = torch.Tensor(cv2.imread(path_image1))
        image2 = torch.Tensor(cv2.imread(path_image2))

        if self.use_random_rotation:
            # TODO:Some rotations lead to changing the colors of the images...Could this lead to problems?
            #  They are: 0,1(when we flip horizontally) or 1,1(both flips applied)
            # Simulate a random rotation of the images with degrees values form the list [0,90,180,270]
            flip_vertically = torch.randint(0, 2, (1,)).item()
            flip_horizontally = torch.randint(0, 2, (1,)).item()

            augmentation = transforms.Compose([
                transforms.RandomVerticalFlip(flip_vertically),
                transforms.RandomHorizontalFlip(flip_horizontally)
            ])
            image1 = augmentation(image1)
            image2 = augmentation(image2)

        for transform in self.transformers:
            image1 = transform(image1)
            image2 = transform(image2)

        return image1, image2, months_between


if __name__ == '__main__':
    dataset = MosaicDataset(dataset_folder="Homework Dataset", use_random_rotation=True)
    for i in range(1000, 1001):
        image1, image2, months_between = dataset.__getitem__(i)

        cv2.imshow("Im1", np.array(image1, dtype=np.uint8))
        cv2.imshow("Im2", np.array(image2, dtype=np.uint8))
        # print("Mb: ", months_between)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
