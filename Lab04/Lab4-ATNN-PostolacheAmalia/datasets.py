import torch
import os
import re
import cv2
from torch.utils.data import Dataset
from torch import Tensor

class CustomDataset(Dataset):
    def __init__(self, folder_path, transforms=None):
        self.folderPath = folder_path
        self.transforms = transforms
        self.imagePaths = {}
        self.results = []
        Lfolders = [f for f in os.listdir(folder_path)]
        for Lfolder in Lfolders:
            imageList = [f for f in os.listdir(folder_path + '\\' + Lfolder + '\\images')]
            self.imagePaths[Lfolder] = imageList

        currentL = 0
        pattern = r"2018_(\d{2})"
        for idx in self.imagePaths.keys():
            noOfImages = len(self.imagePaths[idx])
            for i in range(0, noOfImages):
                for j in range(i+1, noOfImages):
                    start_image = os.path.basename(self.imagePaths[idx][i])
                    end_image = os.path.basename(self.imagePaths[idx][j])
                    match1 = re.search(pattern, start_image)
                    match2 = re.search(pattern, end_image)
                    time_skip = 0
                    if match1 and match2:
                        time_skip = abs(int(match2.group(1)) - int(match1.group(1)))
                    resulted_tuple = (start_image, end_image, time_skip)
                    self.results.append(resulted_tuple)
            currentL += 1
    
    def __len__(self):
        return len(self.results)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, int]:
        start_image, end_image, time_skip = self.results[idx]
        if self.transforms:
            start_image = self.transforms(start_image)
            end_image = self.transforms(end_image)
        return start_image, end_image, time_skip
