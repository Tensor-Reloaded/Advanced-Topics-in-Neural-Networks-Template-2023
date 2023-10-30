import torch
from torchvision import transforms
from torch.utils.data import Dataset
import os
from PIL import Image
import random

__all__ = ['ImageDataset']

def make_permutations(folder_path):
    file_list = os.listdir(folder_path)
    image_files = [f for f in file_list if f.lower().endswith(('.tif'))]

    months = []
    for image in image_files:
        split_path = image.split('_')
        months.append((int(split_path[2]) - 2018) * 12 + int(split_path[3]))

    permutations = []
    for idx, image in enumerate(image_files):
        for idx2, new_image in enumerate(image_files[idx + 1:]):
            permutations.append(
                [folder_path + './' + image, folder_path + './' + new_image, months[idx2] - months[idx] + 1])

    return permutations


# Custom dataset class
class ImageDataset(Dataset):
    transform = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation((-30, 30))
    ]
    )

    def __init__(self, dataset_path, images_transforms=None):
        self.images_transforms = images_transforms if images_transforms is not None else self.transform

        self.permutations = []
        folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
        for folder in folders:
            self.permutations += make_permutations(os.path.join(dataset_path,folder) + '/images')

        random.shuffle(self.permutations)
        # Normalize features
        #self.features = torch.tensor(self.features, dtype=torch.float32)
        #self.features = (self.features - self.features.mean(dim=0)) / self.features.std(dim=0)
        #self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.permutations)

    def __getitem__(self, idx):
        #features = self.features[idx]  # Don't transform the original features
        #labels = self.labels[idx]  # Don't transform the original labels
        #for transform in self.feature_transforms:
         #   features = transform(features)
        #for transform in self.label_transforms:
         #   labels = transform(labels)
        image1 = Image.open(self.permutations[idx][0])
        image2 = Image.open(self.permutations[idx][1])

        common_seed = random.randint(0,100)
        random.seed(common_seed)
        input = self.images_transforms(image1)
        output = self.images_transforms(image2)

        value = torch.tensor([self.permutations[idx][2]], dtype= torch.float)
        input = input.flatten()
        output = output.flatten()
        input = torch.cat((input, value), dim=0)
        return input, output


