import os
import torch
from torch import tensor
from torch.utils.data import Dataset
import pandas as pd
import cv2


class PhotoDataset(Dataset):
    def __init__(self, dataset_folder_location):
        self.fileDictionary = {}
        self.photoLocations = []
        self.monthDifference = []
        subfolders = [f for f in os.listdir(dataset_folder_location)]
        for folder in subfolders:
            files = [f for f in os.listdir(dataset_folder_location + '\\' + folder + '\\images')]
            self.fileDictionary[folder] = files
        for i in self.fileDictionary.keys():
            baseLocation = dataset_folder_location + '\\' + i + '\\images\\'
            for j in range(len(self.fileDictionary[i])):
                for ceva in range(j + 1, len(self.fileDictionary[i])):
                    diferenta = ceva - j
                    self.monthDifference.append(diferenta)
                    self.photoLocations.append(((baseLocation + self.fileDictionary[i][j]), (baseLocation + self.fileDictionary[i][ceva])))

    def __len__(self):
        return len(self.photoLocations)

    def __getitem__(self, index):
        return torch.flatten(tensor(cv2.imread(self.photoLocations[index][0]))), torch.flatten(tensor(cv2.imread(self.photoLocations[index][1]))).view(-1), self.monthDifference[index]
