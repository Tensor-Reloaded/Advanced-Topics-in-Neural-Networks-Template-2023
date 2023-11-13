import os
import torch
from torch import tensor
from torch.utils.data import Dataset
import pandas as pd
import cv2


class MyDataset(Dataset):
    def __init__(self, dataset_folder_location):
        self.dict = {}
        self.photo = []
        self.monthDiff = []
        subfolders = [f for f in os.listdir(dataset_folder_location)]
        for folder in subfolders:
            files = [f for f in os.listdir(dataset_folder_location + '\\' + folder + '\\images')]
            self.dict[folder] = files
        for i in self.dict.keys():
            path = dataset_folder_location + '\\' + i + '\\images\\'
            for j in range(len(self.dict[i])):
                for k in range(j + 1, len(self.dict[i])):
                    self.photo.append([path + self.dict[i][j], path + self.dict[i][k]])
                    self.photo.append(((path + self.dict[i][j]), (path + self.dict[i][k])))
    def __len__(self):
        return len(self.photo)

    def __getitem__(self, index):
        return torch.flatten(tensor(cv2.imread(self.photo[index][0]))), torch.flatten(tensor(cv2.imread(self.photo[index][1]))).view(-1), self.monthDiff[index]