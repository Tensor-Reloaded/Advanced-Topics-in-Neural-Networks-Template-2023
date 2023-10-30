import torch
from torch import nn, optim
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms

from datasets import MetaDataset
from model import Model

transform = transforms.Compose([
    torchvision.transforms.Grayscale(),
    torchvision.transforms.Lambda(lambda x: x.view(-1)) # flatten
])


dataset = MetaDataset('C:/School/Sem1/CARN/Advanced-Topics-in-Neural-Networks-Template-2023/Lab04/Homework Dataset/', transform)

train_size = int(0.7 * len(dataset))
val_size = (len(dataset) - train_size) // 2
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

batch_size = 64

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


dataiter = iter(train_loader)
first_item = next(dataiter)
img1, img2, months_between = first_item

print(img1.size(), img2.size(), months_between.size())

model = Model(image_size=16384, hidden_size=128)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

n_epochs = 10 
model.run(train_loader, validation_loader, criterion, optimizer, n_epochs)
