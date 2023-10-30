from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
from torchvision import transforms
from torchvision.transforms import v2

from models import GlobalImageMLP
from transforms import *
from torch.utils.data import DataLoader

from data_reader import DataReader
from datasets import GlobalImageDataset


def create_graphs(epoch_list, training_loss_list, validation_loss_list):
    plt.plot(epoch_list, training_loss_list, label = "Training")
    plt.plot(epoch_list, validation_loss_list, label="Validation")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def run(n, model, train_loader, validation_loader, device=None):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epoch_list = list(range(1,n+1))
    training_loss_list = []
    validation_loss_list = []
    if device is not None:
        model = model.to(device)

    for epoch in range(n):
        # Training step
        model.train()
        loss_per_epoch = 0
        for features, labels, time_skip in train_loader:
            if device is not None:
                features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_per_epoch += loss.item()
        training_loss_list.append(loss_per_epoch / len(train_loader))
        print(f'[Training] Epoch {epoch + 1}/{n}, Loss: {loss_per_epoch / len(train_loader)}\n')

        # Validation step
        model.eval()
        loss_per_epoch = 0
        with torch.no_grad():
            for features, labels, time_skip in validation_loader:
                if device is not None:
                    features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                loss_per_epoch += loss.item()
        validation_loss_list.append(loss_per_epoch / len(validation_loader))
        print(f'[Validation] Epoch {epoch + 1}/{n}, Loss: {loss_per_epoch / len(validation_loader)}\n')

    # Testing step
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for features, labels, time_skip in test_loader:
            if device is not None:
                features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
    print(f'Test Loss: {test_loss}')
    create_graphs(epoch_list, training_loss_list, validation_loss_list)


if __name__ == '__main__':

    # data_reader = DataReader(folder_path, 6)
    # data_reader.read_and_convert_images()
    # data_reader.create_dataset()
    #
    # # print(type(data_reader.dataset[0][0]),type(data_reader.dataset[0][1]),type(data_reader.dataset[0][2]))
    #
    #
    # #Image.fromarray(v2.Grayscale(data_reader.inputs[0]).numpy()).show()
    #
    #data_transforms = [v2.Grayscale(), transforms.ToTensor()]
    # img = data_reader.dataset[0][0]
    # for transform in data_transforms:
    #     img = transform(img)
    #
    # print(img.flatten())
    # print(len(data_reader.dataset))

    folder_path = "Homework Dataset"

    data_transforms = [Padding(), RandomGaussianBlur(), RandomRotation(), v2.Grayscale(), transforms.ToTensor()]
    total_dataset = GlobalImageDataset(folder_path, data_transforms)
    input_dim = output_dim = len(total_dataset.__getitem__(0)[0])

    # Split the dataset in 3 datasets for training, validation and testing
    train_dataset, validate_dataset,test_dataset = torch.utils.data.random_split(total_dataset, [0.7, 0.15, 0.15])

    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    validation_loader = DataLoader(validate_dataset, batch_size=10, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

    model = GlobalImageMLP(input_dim, output_dim)
    # criterion = nn.BCELoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)

    run(10, model, train_loader, validation_loader)


    # model.train()
    # for features, labels, time_skip in validation_loader:
    #     optimizer.zero_grad()
    #     outputs = model(features)
    #     loss = criterion(outputs, labels)
    #     loss.backward()
    #     optimizer.step()
    #     print(loss.item())


    #print(validate_dataset.__len__())




    # total_dataset.dataset[0][0].show()
    # total_dataset.dataset[0][1].show()

    # inp, out = total_dataset.__getitem__(0)
    # inp.show()
    # out.show()
    # print(out.shape)



