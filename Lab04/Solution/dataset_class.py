import os
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import random
from torchvision import transforms

class ImageDataset(Dataset):
    #setting vars for inputted values
    def __init__(self, folder_path, time_skip, transform = None):
        self.folder_path = folder_path
        self.time_skip = time_skip
        self.image_paths = self._get_image_paths()
        self.dataset_size = len(self.image_paths) - time_skip
        self.transform = transform

    def _get_image_paths(self):
        image_paths = []
        for root, _, files in os.walk(self.folder_path):
            for file in files:
                if file.endswith('.tif'):
                    image_paths.append(os.path.join(root, file))
        return sorted(image_paths)

    def __len__(self):
        return self.dataset_size
    
    def __getitem__(self, index):
        start_image_path = self.image_paths[index]
        end_image_path = self.image_paths[index+self.time_skip]

        start_image = Image.open(start_image_path)
        end_image = Image.open(end_image_path)

        if self.transform:
            start_image = self.transform(start_image)
            end_image = self.transform(end_image)

        return start_image, end_image, self.time_skip
    
transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
])


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.tanh(self.conv3(x))  # Using Tanh for output to ensure pixel values between -1 and 1 for image generation
        return x

def train(model, device, train_loader, optimizer, criterion):
    model.train()  # Set the model to train mode
    running_loss = 0.0

    for batch_idx, (inputs, targets, _) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()  # Zero the gradients

        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, targets)  # Calculate the loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        running_loss += loss.item()

    return running_loss / len(train_loader)

def val(model, device, val_loader, criterion):
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0

    with torch.no_grad():
        for inputs, targets, _ in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, targets)  # Calculate the loss

            val_loss += loss.item()

    return val_loss / len(val_loader)

def run(model, device, train_loader, val_loader, optimizer, criterion, n_epochs):
    train_losses = []
    val_losses = []

    for epoch in range(n_epochs):
        print(f"Epoch {epoch + 1}/{n_epochs}")

        # Train the model
        train_loss = train(model, device, train_loader, optimizer, criterion)
        print(f"Train Loss: {train_loss:.4f}")
        train_losses.append(train_loss)

        # Validate the model
        val_loss = val(model, device, val_loader, criterion)
        print(f"Validation Loss: {val_loss:.4f}")
        val_losses.append(val_loss)

    return train_losses, val_losses

dataset = ImageDataset(folder_path=r"C:\Users\flori\Desktop\AtnnGit\Advanced-Topics-in-Neural-Networks-2023\Lab04\Homework Dataset", time_skip=4, transform = transform)

#Example test for dataset class ex1
print("Dataset size:", len(dataset))
sample_start_image, sample_end_image, sample_time_skip = dataset[0]
print("Sample images shape:", sample_start_image.shape, sample_end_image.shape)
print("Time skip:", sample_time_skip)

train_size = 0.7
validation_size = 0.15
test_size = 0.15

indices =  list(range(len(dataset)))

train_split = int(train_size * len(dataset))
validation_split = int(validation_size * len(dataset))
test_split = len(dataset) - train_split - validation_split

random.seed(42)
random.shuffle(indices)

train_indices = indices[:train_split]
validation_indices = indices[train_split:train_split+validation_split]
test_indices = indices[train_split + validation_split:]

train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
validation_sampler = torch.utils.data.SubsetRandomSampler(validation_indices)
test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)

train_loader = DataLoader(dataset, batch_size=64, sampler=train_sampler)
validation_loader = DataLoader(dataset, batch_size=64, sampler=validation_sampler)
test_loader = DataLoader(dataset, batch_size=64, sampler=test_sampler)

print(f"Train indices length: {len(train_indices)}")
print(f"Validation indices length: {len(validation_indices)}")
print(f"Test indices length: {len(test_indices)}")

train_percent = 100*len(train_indices)/len(dataset)
validation_percent = 100*len(validation_indices)/len(dataset)
test_percent = 100*len(test_indices)/len(dataset)

# Print lengths of DataLoader for each set
print(f"Train DataLoader length: {len(train_loader)} = {train_percent}%")
print(f"Validation DataLoader length: {len(validation_loader)} = {validation_percent}%")
print(f"Test DataLoader length: {len(test_loader)} = {test_percent}%")
print(f"Total percentage: {train_percent+validation_percent+test_percent}")

# Define device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the model
model = Model().to(device)

# Define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Run training and validation
n_epochs = 10  # Specify the number of epochs
train_losses, val_losses = run(model, device, train_loader, validation_loader, optimizer, criterion, n_epochs)

# Plotting the loss
import matplotlib.pyplot as plt

plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()