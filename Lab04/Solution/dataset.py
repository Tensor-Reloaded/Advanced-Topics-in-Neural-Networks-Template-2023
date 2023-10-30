import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import cv2
from PIL import Image
__all__ = ['LandscapeDataset']

# Custom dataset class
class LandscapeDataset(Dataset):
    def __init__(self, root,fold_splitter=None, transforms=None):
        self.data =[]
        self.transforms=transforms
        for filename in os.listdir(root):
            #print(filename)
            if fold_splitter==None or filename in fold_splitter:
                path=os.path.join(root,filename)
                path=path+'/images'
                aux=[]
                for file in os.listdir(path):
                    img = cv2.imread(os.path.join(path,file))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    year=int(file.split('_')[2])
                    month=int(file.split('_')[3])
                    aux.append((img,year,month))
                aux=sorted(aux,key=lambda x:(x[1],x[2]))
                for i in range(0,len(aux)):
                    for j in range(i+1,len(aux)):
                        self.data.append((aux[i][0],aux[j][0],j-i))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_input = self.data[idx][0] 
        label=self.data[idx][1]
        time=self.data[idx][2]
        #print(img_input.shape)
        if self.transforms is not None:
            transformed = self.transforms(image=img_input, label=label)
            img_input=transformed['image']
            label=transformed['label']
        return img_input.reshape(-1),label.reshape(-1),time
