import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import *
from models import *
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transforms import *

# Generate train test split indices
total_dataset = WineDataset(dataset_file="winequality-red.csv")

print("Number of samples", len(total_dataset))
print("Number of features", len(total_dataset.features.shape))
print("Number of classes", total_dataset.labels.unique())


feature_transforms = [WineFeatureGaussianNoise(0, 0.1)] # Try to compare with no augmentation
label_transforms = [OneHot(total_dataset.labels.unique())] # [3, 4, 5, 6, 7, 8]

total_dataset = WineDataset(dataset_file="winequality-red.csv", feature_transforms=feature_transforms, label_transforms=label_transforms)

# Split the dataset into training and testing sets
train_dataset, test_dataset = torch.utils.data.random_split(total_dataset, [0.7, 0.3])

# Alternatively, you can use the train_test_split function from sklearn and initialize the WineDataset with the split indices
# train_indices, test_indices = train_test_split(np.arange(len(total_dataset)), test_size=0.3, random_state=42)
# train_dataset = WineDataset(dataset_file="winequality-red.csv", split_indices=train_indices, feature_transforms=feature_transforms, label_transforms=label_transforms)
# test_dataset = WineDataset(dataset_file="winequality-red.csv", split_indices=test_indices, feature_transforms=feature_transforms, label_transforms=label_transforms)


# Create instances of DataLoader for training and testing
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Instantiate the model, loss function and optimizer
model = WineQualityMLP(input_dim=11, \
    output_dim = len(total_dataset.labels.unique()), \
    output_activation=nn.Softmax(dim=1))
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 500
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for features, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader)}')

# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for features, labels in test_loader:
        outputs = model(features)
        total += labels.size(0)
        correct += (outputs.argmax(dim=1) == labels.argmax(dim=1)).sum().item()

print(f'Accuracy: {100 * correct / total}%')
