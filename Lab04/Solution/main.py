import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import *
from model import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from transform import *

torch.device("cpu")

feature_transforms = [Resize(100)]
label_transforms = [Resize(100)]
combined_random_transforms = [HorizontalFlip(0.5), RandomRotate()]

total_dataset = ImageDataset(dataset_file="./data/images.csv",
                             feature_transforms=feature_transforms,
                             label_transforms=label_transforms,
                             combined_random_transforms=combined_random_transforms)

print("Number of samples", len(total_dataset))
print("Number of features", len(total_dataset.features.shape))

train_set, valid_set, test_set = torch.utils.data.random_split(total_dataset, [0.7, 0.15, 0.15])


train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=32, shuffle=False)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

start_image, time_skip, end_image = next(iter(train_loader))
features_size = start_image.size(1) + 1
labels_size = end_image.size(1)

model = ImageMLP(input_dim=features_size, output_dim=labels_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epoch_array = []
valid_loss_array = []
train_loss_array = []

num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    pbar = tqdm(total=len(train_loader), desc="Training", dynamic_ncols=True)
    for features, time_skip, labels in train_loader:
        optimizer.zero_grad()
        full_features = torch.cat((features, time_skip.unsqueeze(1)), dim=1)
        outputs = model(full_features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        pbar.set_postfix({'Loss': loss.item()})
        pbar.update()
    train_loss_array.append(total_loss / len(train_loader))

    model.eval()
    total_loss = 0
    pbar = tqdm(total=len(valid_loader), desc="Validation", dynamic_ncols=True)
    for features, time_skip, labels in valid_loader:
        full_features = torch.cat((features, time_skip.unsqueeze(1)), dim=1)
        outputs = model(full_features)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        pbar.set_postfix({'Loss': loss.item()})
        pbar.update()

    print("Epoch: {}, Validation Loss: {}".format(epoch, total_loss / len(valid_loader)))
    epoch_array.append(epoch)
    valid_loss_array.append(total_loss / len(valid_loader))

# Test
model.eval()
mse = 0


def mean_squared_error(target, predicted):
    return torch.mean((predicted - target) ** 2)


with torch.no_grad():
    for features, time_skip, labels in test_loader:
        full_features = torch.cat((features, time_skip.unsqueeze(1)), dim=1)
        outputs = model(full_features)

        # Calculate the MSE between predicted and target images
        mse += mean_squared_error(labels, outputs)

# Calculate the average MSE across all test samples
mse /= len(test_loader)
print(f'Mean Squared Error: {mse}')
#####################################################
# Mean Squared Error (50 epochs): 4.7666219415987143e-07
#####################################################

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

plt.plot(epoch_array, train_loss_array)

plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss vs Epoch")
plt.show()


plt.plot(epoch_array, valid_loss_array)

plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.title("Validation Loss vs Epoch")
plt.show()
