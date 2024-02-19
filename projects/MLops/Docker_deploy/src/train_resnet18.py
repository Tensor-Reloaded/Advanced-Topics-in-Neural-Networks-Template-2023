import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import time
import os

BATCH_SIZE = 128
LEARNING_RATE = 0.01
STEP_SIZE = 20
GAMMA = 0.1

def download_cifar100():
    data_path = "../data"
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    datasets.CIFAR100(root=data_path, train=True, download=True)
    datasets.CIFAR100(root=data_path, train=False, download=True)

def main():
    download_cifar100()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    trainset = datasets.CIFAR100(root="../data", train=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    testset = datasets.CIFAR100(root="../data", train=False, transform=transform)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 100)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA) 

    start_time = time.time()

    model.train()
    for epoch in range(10000): 
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        if time.time() - start_time > 30 * 60:  # 30 minutes
            break

        scheduler.step()

        if epoch % 1000 == 0:
            print(f"Epoch {epoch} loss: {loss.item()}")

    model.eval()

    torch.save(model.state_dict(), "../resnet18_cifar100_final.pth")

    input_data = torch.randn(1, 3, 32, 32).to(device)
    torch.onnx.export(
        model,
        input_data,
        "../resnet18_cifar100.onnx",
        verbose=True,
        do_constant_folding=True,
        input_names=["modelInput"],
        output_names=["modelOutput"],
        dynamic_axes={"modelInput": {0: "batch_size"}, "modelOutput": {0: "batch_size"}},
    )

if __name__ == "__main__":
    main()