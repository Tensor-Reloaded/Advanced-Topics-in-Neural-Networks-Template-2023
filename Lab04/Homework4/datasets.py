import cv2 as cv
import os
import torch
from torch.utils.data import Dataset
import pandas as pd

__all__ = ["ImageDataset"]

class ImageDataset(Dataset):
    def __init__(self, dataset_file, split_indices=None, feature_transforms=None):
        self.feature_transforms = feature_transforms if feature_transforms is not None else []

        data = pd.DataFrame(columns=['image1', 'month_diff', 'image2'])

        for folders in os.listdir(dataset_file):

            image_folder = os.path.join(dataset_file,folders,"images")
            image_list = os.listdir(image_folder)

            for i in range(len(image_list)-1):

                #swap from BGR to RGB
                img1 = cv.imread(os.path.join(image_folder,image_list[i]))[:, :, [2, 1, 0]]
                #swap from (3,size,size) to (size,size,3) in order for transformations to be applied and normalize data
                tensor1 = torch.from_numpy(img1).permute(2, 0, 1).float()/255

                for j in range(i+1,len(image_list)):
                    month1, year1 = int(image_list[i].split("_")[3]), int(image_list[i].split("_")[2])
                    month2, year2 = int(image_list[j].split("_")[3]), int(image_list[j].split("_")[2])
                    diff = (year2-year1)*12 + (month2-month1)

                    #swap from BGR to RGB
                    img2 = cv.imread(os.path.join(image_folder,image_list[j]))[:, :, [2, 1, 0]]
                    #swap from (3,size,size) to (size,size,3) in order for transformations to be applied and normalize data
                    tensor2 = torch.from_numpy(img2).permute(2, 0, 1).float()/255
                    data.loc[len(data)]= {'image1': tensor1, 'month_diff': diff, 'image2': tensor2}
        
        if split_indices is None:
            split_indices = torch.arange(len(data))
        data = data.iloc[split_indices]

        self.features = data.drop('image2', axis=1).values
        self.labels = data['image2'].values


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
          img1 = self.features[index][0]
          img2 = self.labels[index]
          for transform in self.feature_transforms:
              img1 = transform(img1)
              img2 = transform(img2)
          #return 1 dimension tensor for images in order to be passed to neuronal network
          return (img1.view(-1),self.features[index][1],img2.view(-1))
