import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import re
from datetime import datetime
from torchvision.transforms import functional as F
import random
import torch

class MetaDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.datasets = []
        for subdir in os.listdir(root_dir):
            if os.path.isdir(os.path.join(root_dir, subdir)):
                self.datasets.append(SubfolderDataset(os.path.join(root_dir, subdir, "images"), transform))

    def __len__(self):
        return sum(len(d) for d in self.datasets)

    def __getitem__(self, idx):
        for dataset in self.datasets:
            if idx < len(dataset):
                return dataset[idx]
            idx -= len(dataset)

class SubfolderDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path

        if transform is not None:
            self.transform = transforms.Compose([
                transforms.ToTensor(), 
                transform
            ])
        else:
            self.transform = transforms.ToTensor()

        image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.tif')])
        self.image_pairs = [(image_files[i], image_files[j]) 
                            for i in range(len(image_files)) 
                            for j in range(i+1, len(image_files))]

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        img1_path, img2_path = self.image_pairs[idx]
        img1 = Image.open(os.path.join(self.folder_path, img1_path))
        img2 = Image.open(os.path.join(self.folder_path, img2_path))

        angle = random.uniform(-30, 30)

        # Apply the same rotation to both images
        img1 = F.rotate(img1, angle)
        img2 = F.rotate(img2, angle)

        img1 = self.transform(img1)
        img2 = self.transform(img2)

        months_between = self.__months_between(img1_path, img2_path)
        months_between = torch.tensor([months_between], dtype=torch.float32)
        return img1, img2, months_between

    def __months_between(self, path1, path2):
        date_string1 = re.findall(r'global_monthly_(\d{4}_\d{2})', path1)[0]
        date_string2 = re.findall(r'global_monthly_(\d{4}_\d{2})', path2)[0]
        date1 = datetime.strptime(date_string1, '%Y_%m')
        date2 = datetime.strptime(date_string2, '%Y_%m')
        return (date2.year - date1.year) * 12 + date2.month - date1.month

