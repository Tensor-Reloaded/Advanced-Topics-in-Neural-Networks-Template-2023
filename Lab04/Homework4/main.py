import torch
from datasets import *
from torch.utils.data import DataLoader
from transforms import *
from models import *
import torch.optim as optim
import matplotlib.pyplot as plt
import time

DEVICE_PARAM = None
N_EPOCHS = 10  
NN_DIM = 3*138*138 #image size after adding padding

start_time = time.time()

image_transforms = [ImageTransform()]

total_dataset = ImageDataset(dataset_file="D:\Facultate\Master1\ACNN\Teme\Homework4\Homework Dataset", feature_transforms=image_transforms)

train_dataset, test_dataset, validation_dataset = torch.utils.data.random_split(total_dataset, [0.7, 0.15, 0.15])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
valid_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)


model = ImagePredictMLP(input_dim=NN_DIM, output_dim = NN_DIM)
if DEVICE_PARAM is not None:
    if "cuda" in DEVICE_PARAM and torch.cuda.is_available() == False:
        DEVICE_PARAM = None
if DEVICE_PARAM is not None:
    model = model.to(DEVICE_PARAM)


criterion = nn.MSELoss()  # Use an appropriate loss function
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Adjust the optimizer and learning rate as needed

# Training and validation functions
def train(device_param):
    model.train()
    total_loss = 0.0

    for img_feature, diff, img_label in train_loader:
        if device_param is not None:
            img_feature, diff, img_label = img_feature.to(device_param), diff.to(device_param), img_label.to(device_param)
        optimizer.zero_grad()
        outputs = model(img_feature,diff)
        loss = criterion(outputs, img_label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)

def val(device_param):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for img_feature, diff, img_label in valid_loader:
            if device_param is not None:
                img_feature, diff, img_label = img_feature.to(device_param), diff.to(device_param), img_label.to(device_param)
            outputs = model(img_feature, diff)
            loss = criterion(outputs, img_label)
            total_loss += loss.item()

    return total_loss / len(valid_loader)

def test(device_param):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for img_feature, diff, img_label in test_loader:
            if device_param is not None:
                img_feature, diff, img_label = img_feature.to(device_param), diff.to(device_param), img_label.to(device_param)
            outputs = model(img_feature, diff)
            loss = criterion(outputs, img_label)
            total_loss += loss.item()

    print(f'Test Loss: {total_loss / len(test_loader)}\n')


def run(n_epochs, device_param=None):
    if device_param is not None:
        if "cuda" in device_param and torch.cuda.is_available() == False:
            device_param = None
    train_losses = []
    val_losses = []

    for epoch in range(n_epochs):
        train_loss = train(device_param)
        val_loss = val(device_param)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch [{epoch+1}/{n_epochs}]: Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    test(device_param)

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, n_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, n_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('metrics.png')


run(N_EPOCHS, DEVICE_PARAM)


end_time = time.time()
print(f"Elapsed time: {end_time-start_time} seconds")
