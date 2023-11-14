import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datasets import WineDataset
from models import WineQualityMLP
from transforms import WineFeatureGaussianNoise, OneHot
from sklearn.model_selection import train_test_split
import wandb

wandb.login(key="044dcd6db8b59791c6b07c29f846735bc94bd5ae")

total_dataset = WineDataset(dataset_file="winequality-red.csv")

feature_transforms = [WineFeatureGaussianNoise(0, 0.1)]
label_transforms = [OneHot(total_dataset.labels.unique())]

total_dataset = WineDataset(dataset_file="winequality-red.csv", feature_transforms=feature_transforms, label_transforms=label_transforms)

train_dataset, test_dataset = train_test_split(total_dataset, test_size=0.3, random_state=42)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = WineQualityMLP(input_dim=11, output_dim=len(total_dataset.labels.unique()), output_activation=nn.Softmax(dim=1))
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

writer = SummaryWriter()

wandb.init(project="wine_quality")

num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total_samples = 0

    for batch_idx, (features, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

        writer.add_scalar('Batch Training Loss', loss.item(), epoch * len(train_loader) + batch_idx)

    epoch_train_loss = total_loss / len(train_loader)
    epoch_train_accuracy = correct / total_samples
    writer.add_scalar('Epoch Training Loss', epoch_train_loss, epoch)
    writer.add_scalar('Epoch Training Accuracy', epoch_train_accuracy, epoch)

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for features, labels in test_loader:
        outputs = model(features)
        _, predicted = torch.max(outputs, 1)

        # Debugging prints
        print("Predicted size:", predicted.size())
        print("Labels size:", labels.size())

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total
writer.add_scalar('Test Accuracy', test_accuracy)

wandb.log({"learning_rate": 0.001, "optimizer": "Adam", "batch_size": 32, "test_accuracy": test_accuracy})

writer.close()
