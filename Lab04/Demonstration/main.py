import torch
import torch.nn as nn
import torch.optim as optim
from datasets import *
from models import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from transforms import *


def compute_model_accuracy2(data_loader, model, classes):
    no_labels = len(classes)
    values = torch.zeros((no_labels,))

    length = 1 / (classes.max() - classes.min())
    for index in range(1, no_labels - 1):
        values[index] = values[index - 1] + length
    values[no_labels - 1] = 1
    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in data_loader:
            outputs = model(features)
            total += labels.size(0)

            for index in range(len(outputs)):
                pred = (values - outputs[index]).abs().argmax() + 3
                target = (values - labels[index]).abs().argmax() + 3
                correct += (pred == target)

    return 100 * correct / total


def ex2():
    # Generate train test split indices
    total_dataset = WineDataset(dataset_file="winequality-red.csv")

    print("Number of samples", len(total_dataset))
    print("Number of features", total_dataset.features.shape[1])
    print("Number of classes", total_dataset.labels.unique())

    feature_transforms = [WineFeatureGaussianNoise(0, 0.1)]
    classes = total_dataset.labels.unique()
    label_transforms = [RegressionEx2(classes)]  # [3, 4, 5, 6, 7, 8]

    total_dataset = WineDataset(dataset_file="winequality-red.csv", \
                                feature_transforms=feature_transforms, \
                                label_transforms=label_transforms)

    # Split the dataset into training and testing sets
    train_dataset, test_dataset = torch.utils.data.random_split(total_dataset, [0.7, 0.3])

    # Alternatively, you can use the train_test_split function from sklearn and initialize the WineDataset with the split indices
    ## import numpy as np
    ## from sklearn.model_selection import train_test_split
    ## train_indices, test_indices = train_test_split(np.arange(len(total_dataset)), test_size=0.3, random_state=42)
    ## train_dataset = WineDataset(dataset_file="winequality-red.csv", split_indices=train_indices, feature_transforms=feature_transforms, label_transforms=label_transforms)
    ## test_dataset = WineDataset(dataset_file="winequality-red.csv", split_indices=test_indices, feature_transforms=feature_transforms, label_transforms=label_transforms)

    # Create instances of DataLoader for training and testing
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Instantiate the model, loss function and optimizer
    model = WineQualityMLP(input_dim=11, \
                           output_dim=1, \
                           output_activation=nn.Identity())
    # model = model.to('mps')
    # TODO:How does Binary Cross Entropy make sense for 6 classes to classify?
    #  I assume from what i found online that BCE still does the generalized cross entropy but expects the labels one hot?
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(total=len(train_loader), desc="Training", dynamic_ncols=True)
        for features, labels in train_loader:
            # features, labels = features.to('mps'), labels.to('mps')
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            pbar.set_postfix({'Loss': loss.item()})
            pbar.update()

        pbar.close()
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}')

        print(f"Training accuracy: {compute_model_accuracy2(train_loader, model, classes)}")
        print(f"Test accuracy: {compute_model_accuracy2(test_loader, model, classes)}")


def compute_model_accuracy3(data_loader, model, threshold: int, classes):
    if threshold not in classes:
        print("Threshold has to be associated with an available value")
        exit(0)

    no_labels = len(classes)
    values = torch.zeros((no_labels,))

    length = 1 / (classes.max() - classes.min())
    for index in range(1, no_labels - 1):
        values[index] = values[index - 1] + length
    values[no_labels - 1] = 1

    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in data_loader:
            outputs = torch.squeeze(model(features)) > values[threshold - 3]
            labels_values = labels > 0.5

            correct += labels.size(0) - torch.logical_xor(outputs, labels_values).sum().item()
            total += labels.size(0)

    return 100 * correct / total


def ex3(threshold: int):
    # Generate train test split indices
    total_dataset = WineDataset(dataset_file="winequality-red.csv")

    print("Number of samples", len(total_dataset))
    print("Number of features", total_dataset.features.shape[1])
    print("Number of classes", total_dataset.labels.unique())

    feature_transforms = [WineFeatureGaussianNoise(0, 0.1)]
    label_transforms = [RegressionEx3(threshold)]
    classes = total_dataset.labels.unique()

    total_dataset = WineDataset(dataset_file="winequality-red.csv", \
                                feature_transforms=feature_transforms, \
                                label_transforms=label_transforms)

    # Split the dataset into training and testing sets
    train_dataset, test_dataset = torch.utils.data.random_split(total_dataset, [0.7, 0.3])

    # Alternatively, you can use the train_test_split function from sklearn and initialize the WineDataset with the split indices
    ## import numpy as np
    ## from sklearn.model_selection import train_test_split
    ## train_indices, test_indices = train_test_split(np.arange(len(total_dataset)), test_size=0.3, random_state=42)
    ## train_dataset = WineDataset(dataset_file="winequality-red.csv", split_indices=train_indices, feature_transforms=feature_transforms, label_transforms=label_transforms)
    ## test_dataset = WineDataset(dataset_file="winequality-red.csv", split_indices=test_indices, feature_transforms=feature_transforms, label_transforms=label_transforms)

    # Create instances of DataLoader for training and testing
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Instantiate the model, loss function and optimizer
    model = WineQualityMLP(input_dim=11, \
                           output_dim=1, \
                           output_activation=nn.Identity())
    # model = model.to('mps')
    # TODO:How does Binary Cross Entropy make sense for 6 classes to classify?
    #  I assume from what i found online that BCE still does the generalized cross entropy but expects the labels one hot?
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(total=len(train_loader), desc="Training", dynamic_ncols=True)
        for features, labels in train_loader:
            # features, labels = features.to('mps'), labels.to('mps')
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            pbar.set_postfix({'Loss': loss.item()})
            pbar.update()

        pbar.close()
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}')

        print(f"Training accuracy: {compute_model_accuracy3(train_loader, model, threshold, classes)}")
        print(f"Test accuracy: {compute_model_accuracy3(test_loader, model, threshold, classes)}")


# ex2()
ex3(7)
