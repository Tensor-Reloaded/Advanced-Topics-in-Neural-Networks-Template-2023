import torch
import torch.nn as nn
import torch.optim as optim
from dataset import *
from model import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from transform import *

feature_transforms = [Crop(100), Resize(224), HorizontalFlip, Rotate(15)]
label_transforms = [Crop(100), Resize(224), HorizontalFlip, Rotate(15)]

total_dataset = ImageDataset(dataset_file="./data/images.csv")

print("Number of samples", len(total_dataset))
print("Number of features", len(total_dataset.features.shape))

train_set, valid_set, test_set = torch.utils.data.random_split(total_dataset, [0.7, 0.15, 0.15])


train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=32, shuffle=False)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

start_image, time_skip, end_image = next(iter(train_loader))
features_size = start_image.size(1) + 1
labels_size = end_image.size(1)

model = ImageMLP(input_dim=features_size, output_dim=labels_size, output_activation=nn.Softmax(dim=1))
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
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

# Test
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for features, time_skip, labels in test_loader:
        full_features = torch.cat((features, time_skip.unsqueeze(1)), dim=1)
        outputs = model(full_features)
        total += labels.size(0)
        correct += (outputs.argmax(dim=1) == labels.argmax(dim=1)).sum().item()


print(f'Test Accuracy: {100 * correct / total}%')


