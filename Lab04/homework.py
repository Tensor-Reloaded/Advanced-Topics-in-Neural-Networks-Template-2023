import os
import random
import time
from typing import Callable, Optional

import numpy
import torch
from numpy import ndarray
from torch import tensor
from torch.utils.data import Dataset
import pandas as pd
import cv2
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.optim as optim

class photo_dataset(Dataset):
    def __init__(self, dataset_folder_location, transform: Optional[Callable]):
        self.tr = transform
        if self.tr is None:
            self.tr = lambda x: x

        self.file_dict = {}
        self.photos = []
        self.delta = []
        sub_folders = [f for f in os.listdir(dataset_folder_location)]
        for folder in sub_folders:
            files = [f for f in os.listdir(dataset_folder_location + '/' + folder + '/images')]
            self.file_dict[folder] = files
        for folder, content in self.file_dict.items():
            img_folder = dataset_folder_location + '/' + folder + '/images/'
            for image in range(len(content)):
                for future in range(image + 1, len(content)):
                    delta = future - image
                    self.delta.append(delta)
                    self.photos.append(((img_folder + content[image]), (img_folder + content[future])))
        self.length = len(self.photos)

    def bitwise_onehot(self, x: int):
        return torch.Tensor([x & (1 << sh) != 0 for sh in range(8)])

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return (torch.flatten(tensor(self.tr(cv2.imread(self.photos[index][0])))),
                torch.flatten(tensor(self.tr(cv2.imread(self.photos[index][1])))),
                self.bitwise_onehot(self.delta[index]))


class PhotoPredictModel(nn.Module):
    def __init__(self, input_dimensions, output_dimensions):
        super(PhotoPredictModel, self).__init__()
        self.f1 = nn.Linear(input_dimensions, output_dimensions // 4)
        self.f2 = nn.Linear(output_dimensions // 4, output_dimensions // 8)
        self.f3 = nn.Linear(output_dimensions // 8, output_dimensions)
        self.final = nn.Sigmoid()

    def forward(self, x):
        x = x / 255.
        x = torch.relu(self.f1(x))
        x = torch.relu(self.f2(x))
        x = self.final(self.f3(x))
        return 255 * x


if __name__ == '__main__':
    regen = True
    rot = 0
    scale = 1

    def no_xform(img):
        return img

    def xform(img: ndarray):  # function meant to be run in paired calls
        global regen, rot, scale
        if regen:
            rot = random.random() * 360
            scale = random.random() + 1  # 1 to 1.5

        regen = not regen
        h, w = img.shape[:2]
        h = h // 2
        w = w // 2
        m = cv2.getRotationMatrix2D((w, h), rot, scale)
        return cv2.warpAffine(img, m, (h, w))

    complete_dataset = photo_dataset('Homework Dataset', None)
    a = complete_dataset[0]

    train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(complete_dataset, [0.7, 0.15, 0.15])
    model = PhotoPredictModel(input_dimensions=128 * 128 * 3 + 8, output_dimensions=128 * 128 * 3).to('cuda')
    criterion = nn.L1Loss()

    # train
    num_epochs = 100
    epochs = tqdm(range(num_epochs))
    for epoch in epochs:
        model.train()
        split = 2048
        indexes_space = [i for i in range(len(train_dataset))[::split]]
        indexes = [[a + offset for a in indexes_space if a + offset < len(train_dataset)] for offset in range(split)]
        subsets = [Subset(train_dataset, ind) for ind in indexes]
        x = 0
        for subset in subsets:
            train_loader = DataLoader(subset, batch_size=1024, shuffle=True, num_workers=4)
            for begin_photo, end_photo, delta in train_loader:
                begin_photo, end_photo, delta = begin_photo.to('cuda').float(), end_photo.to('cuda').float(), delta.to('cuda').float()
                temp = torch.cat((begin_photo, delta), dim=1)
                outputs = model(temp)
                loss = criterion(outputs, end_photo)
                loss.backward()
            torch.cuda.empty_cache()
            x += 1
            print('subset done', x)

        # validate
        model.eval()
        split = 1024
        indexes_space = [i for i in range(len(validation_dataset))[::split]]
        indexes = [[a + offset for a in indexes_space if a + offset < len(validation_dataset)] for offset in
                   range(split)]
        subsets = [Subset(train_dataset, ind) for ind in indexes]
        correct = 0
        total = 0
        for val_subset in subsets:
            validation_loader = DataLoader(val_subset, batch_size=1024, shuffle=False, num_workers=2)
            with torch.no_grad():
                for begin_photo, end_photo, delta in validation_loader:
                    begin_photo, end_photo, delta = begin_photo.to('cuda').float(), end_photo.to('cuda').float(), delta.to('cuda').float()
                    temp = torch.cat((begin_photo, delta), dim=1)
                    outputs = model(temp)
                    total += end_photo.size(dim=0)
                    correct += (outputs.argmax(dim=1) == end_photo.argmax(dim=1)).sum().item()
            torch.cuda.empty_cache()
        epochs.set_postfix_str(f"accuracy = {100 * correct / total}%")

    # Evaluation
    model.eval()
    split = 1024
    indexes_space = [i for i in range(len(validation_dataset))[::split]]
    indexes = [[a + offset for a in indexes_space if a + offset < len(validation_dataset)] for offset in
               range(split)]
    subsets = [Subset(train_dataset, ind) for ind in indexes]
    correct = 0
    total = 0
    for val_subset in subsets:
        validation_loader = DataLoader(val_subset, batch_size=1024, shuffle=False, num_workers=2)
        with torch.no_grad():
            for begin_photo, end_photo, delta in validation_loader:
                begin_photo, end_photo, delta = begin_photo.to('cuda').float(), end_photo.to('cuda').float(), delta.to(
                    'cuda').float()
                temp = torch.cat((begin_photo, delta), dim=1)
                outputs = model(temp)
                total += end_photo.size(dim=0)
                correct += (outputs.argmax(dim=1) == end_photo.argmax(dim=1)).sum().item()
        torch.cuda.empty_cache()
    epochs.set_postfix_str(f"accuracy = {100 * correct / total}%")