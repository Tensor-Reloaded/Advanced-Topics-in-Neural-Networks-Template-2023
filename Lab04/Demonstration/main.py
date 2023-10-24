import torch
import torch.nn as nn
import torch.optim as optim
from datasets import *
from models import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from transforms import *

# Generate train test split indices
total_dataset = WineDataset(dataset_file="winequality-red.csv")

print("Number of samples", len(total_dataset))
print("Number of features", len(total_dataset.features.shape))
print("Number of classes", total_dataset.labels.unique())

feature_transforms = [WineFeatureGaussianNoise(0, 0.1)]
label_transforms = [OneHot(total_dataset.labels.unique())] # [3, 4, 5, 6, 7, 8]

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
    output_dim = len(total_dataset.labels.unique()), \
    output_activation=nn.Softmax(dim=1))
# model = model.to('mps')
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

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
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader)}\n')

# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for features, labels in test_loader:
        outputs = model(features)
        total += labels.size(0)
        correct += (outputs.argmax(dim=1) == labels.argmax(dim=1)).sum().item()
        
        

print(f'Test Accuracy: {100 * correct / total}%')
