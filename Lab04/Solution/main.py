import matplotlib.pyplot as plt
import torch.cuda
import os
import matplotlib.pyplot as plt
from torchvision.transforms.v2 import RandomRotation

from Solution.ImageDataset import ImageDataset

def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
        # For multi-gpu workstations, PyTorch will use the first available GPU (cuda:0), unless specified otherwise
        # (cuda:1).
    if torch.backends.mps.is_available():
        return torch.device('mos')
    return torch.device('cpu')


if __name__ == '__main__':
    dataset_folder = os.path.abspath("../HomeWork Dataset")
    transforms = [RandomRotation(degrees=360)]
    img_dataset = ImageDataset(dataset_folder, r"./homework_dataset.csv", get_default_device(), transforms)
    print(len(img_dataset))
    print(img_dataset[0][0].shape)
    print(img_dataset[0][1])
    f, axarr = plt.subplots(2, 1)
    axarr[0].imshow(img_dataset[0][0])
    axarr[1].imshow(img_dataset[0][2])
    plt.show()
