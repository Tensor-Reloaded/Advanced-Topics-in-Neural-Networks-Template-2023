import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import *
from mlp import *
from training_functions import *

my_dataset = CustomDataset("C:/Users/ami24/OneDrive/Desktop/MAIO/ATNN/Advanced-Topics-in-Neural-Networks-Template-2023/Lab04/Homework Dataset")
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(my_dataset, [0.7, 0.15, 0.15])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = CustomMLP(inputDimensions=49153, outputDimensions=49152)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_loss_means, val_loss_means = run(model, train_dataset, val_dataset, 10, criterion, optimizer)

graph("Training loss means", train_loss_means)
graph("Validation loss means", val_loss_means)
