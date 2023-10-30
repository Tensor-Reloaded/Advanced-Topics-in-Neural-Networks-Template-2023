import torchvision
from datasets import *
from pathlib import Path
from torchvision.transforms import RandomRotation, RandomHorizontalFlip, RandomVerticalFlip, GaussianBlur, \
    ElasticTransform
from torch.utils.data import DataLoader
from torch import nn
import torch.optim as optim
from model import BuildingsModel
import training
import matplotlib.pyplot as plt
import numpy as np


def display_one_pair_from_dataset(index: int):
    transform = torchvision.transforms.ToPILImage()
    img1 = transform(total_dataset[index][0])
    img1.show()

    img2 = transform(total_dataset[index][1])
    img2.show()


def display_loss_graph(x1, y1, y2):
    x1points = np.array(range(x1))
    y1points = y1
    y2points = y2

    plt.plot(x1points, y1points, label="Training loss")
    plt.plot(x1points, y2points, label="Validation loss")

    plt.xlabel('Epochs')
    plt.ylabel('Loss values')
    plt.title("Loss values for training and validation instances")

    plt.legend()

    plt.show()


if __name__ == '__main__':
    # 1
    # Upload the dataset (train, validation, test)
    # For this approach, the dataset should be stored in the same folder as the other components which processes the
    # dataset
    input_dir = Path.cwd()  # Dynamic reference to a directory where currently a process is running

    total_dataset = BuildingsDataset(folder_path=input_dir, feature_transforms=[RandomRotation(degrees=(0, 30)),
                                                                                RandomHorizontalFlip(p=0.5),
                                                                                RandomVerticalFlip(p=0.5),
                                                                                GaussianBlur(kernel_size=(5, 9),
                                                                                             sigma=(0.1, 5.)),
                                                                                ElasticTransform(alpha=250.0)])
    # folder_path parameter represents the path where the dataset is stored

    # 2
    # Dataset is already shuffled in constructor
    # Split the dataset into training, validation and testing sets
    train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(total_dataset, [0.7, 0.15, 0.15])

    # Even if instances are already shuffled, I'll shuffle again the training dataset to assure diversity of instances
    train_data = DataLoader(train_dataset, batch_size=32, shuffle=True)
    validation_data = DataLoader(validation_dataset, batch_size=32, shuffle=False)
    test_data = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 3
    # Instantiate the model, loss function and optimizer
    buildingsModel = BuildingsModel(shape_of_image=(3, 128, 128))

    loss_function = nn.MSELoss()
    optimizer = optim.Adam(buildingsModel.parameters(), lr=0.001)

    number_of_epochs = 7

    train_loss_during_epochs, validation_loss_during_epochs = training.run(number_of_epochs, buildingsModel,
                                                                           train_data, validation_data, test_data,
                                                                           loss_function, optimizer)

    display_loss_graph(number_of_epochs, train_loss_during_epochs, validation_loss_during_epochs)
