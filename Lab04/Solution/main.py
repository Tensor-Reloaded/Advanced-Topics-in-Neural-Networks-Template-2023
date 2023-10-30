import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from Lab04.Solution.dataset import Dataset
from Lab04.Solution.model import ImageModel
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),  # Example random augmentation
    transforms.RandomRotation(30),  # Bonus random rotation augmentation
])

dataset = Dataset(root_dir="../Homework Dataset", transform=transform)
train_dataset, val_dataset, test_dataset= torch.utils.data.random_split(dataset, [0.7, 0.15, 0.15])

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


def train(model, optimizer, criterion, train_loader, device):
    model.train()
    total_loss = 0

    for inputs, targets, _ in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)

def val(model, criterion, val_loader, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for inputs, targets, _ in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

    return total_loss / len(val_loader)

def run(model, optimizer, criterion, train_loader, val_loader, device, num_epochs):
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        train_loss = train(model, optimizer, criterion, train_loader, device)
        val_loss = val(model, criterion, val_loader, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    return train_losses, val_losses


model = ImageModel(input_dim=49153,
                   output_dim=49152,
                   output_activation=nn.Softmax(dim=1))

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-2)
device = 'cpu'
num_epochs = 10

train_losses, val_losses = run(model, optimizer, criterion, train_loader, val_loader, device, num_epochs)