import torch
import os
import cv2
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from PhotoDataset import PhotoDataset
from PhotoMLP import PhotoMLP
import torch.nn as nn
import torch.optim as optim

if __name__ == '__main__':
    complete_dataset = PhotoDataset('Homework Dataset')
    train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(complete_dataset, [0.7, 0.15, 0.15])
    model = PhotoMLP(inputDimensions=49153, outputDimensions=49152).to('cuda')
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        subsetIndices = []
        train_loader_subsets = []
        for i in range(10):
            temp = []
            for j in range(len(train_dataset) * i // 10, len(train_dataset) * (i + 1) // 10):
                temp.append(i)
            subsetIndices.append(temp)
        for i in subsetIndices:
            train_loader_subsets.append(Subset(train_dataset, i))
        pbar = tqdm(total=len(train_dataset), desc="Training", dynamic_ncols=True)
        for baby_train_dataset in train_loader_subsets:
            train_loader = DataLoader(baby_train_dataset, batch_size=2, shuffle=True, num_workers=4)
            for firstPhotos, secondPhotos, monthsCount in train_loader:
                firstPhotos, secondPhotos, monthsCount = firstPhotos.to('cuda').float(), secondPhotos.to('cuda').float(), monthsCount.to('cuda').float()
                optimizer.zero_grad()
                if str(monthsCount.shape) == 'torch.Size([2])':
                    temp = torch.cat((firstPhotos, monthsCount.view(2, 1)), dim=1)
                else:
                    temp = torch.cat((firstPhotos.T, monthsCount.view(1, 1)))
                    temp = temp.T
                outputs = model(temp)
                loss = criterion(outputs, secondPhotos)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                pbar.set_postfix({'Loss': loss.item()})
                pbar.update()
            torch.cuda.empty_cache()
        pbar.close()
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}\n')

        # validare
        if epoch % 10 == 0:
            subsetIndices = []
            validation_loader_subsets = []
            for i in range(100):
                temp = []
                for j in range(len(validation_dataset) * i // 100, len(validation_dataset) * (i + 1) // 100):
                    temp.append(i)
                subsetIndices.append(temp)
            for i in subsetIndices:
                validation_loader_subsets.append(Subset(validation_dataset, i))
            for baby_validation_dataset in validation_loader_subsets:
                validation_loader = DataLoader(baby_validation_dataset, batch_size=2, shuffle=False, num_workers=2)
                model.to('cuda')
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for firstPhotos, secondPhotos, monthsCount in validation_loader:
                        firstPhotos, secondPhotos, monthsCount = firstPhotos.to('cuda').float(), secondPhotos.to(
                            'cuda').float(), monthsCount.to('cuda').float()
                        if str(monthsCount.shape) == 'torch.Size([2])':
                            temp = torch.cat((firstPhotos, monthsCount.view(2, 1)), dim=1)
                        else:
                            temp = torch.cat((firstPhotos.T, monthsCount.view(1, 1)))
                            temp = temp.T
                        outputs = model(temp)
                        total += secondPhotos.size(0)
                        correct += (outputs.argmax(dim=1) == secondPhotos.argmax(dim=1)).sum().item()
                print(f'Validation Accuracy at epoch {epoch}: {100 * correct / total}%')
                torch.cuda.empty_cache()
    # Evaluation
    model.to('cuda')
    model.eval()
    correct = 0
    total = 0
    subsetIndices = []
    test_loader_subsets = []
    for i in range(100):
        temp = []
        for j in range(len(test_dataset) * i // 100, len(test_dataset) * (i + 1) // 100):
            temp.append(i)
        subsetIndices.append(temp)
    for i in subsetIndices:
        test_loader_subsets.append(Subset(validation_dataset, i))
    for baby_test_dataset in test_loader_subsets:
        test_loader = DataLoader(baby_test_dataset, batch_size=2, shuffle=False, num_workers=2)
        with torch.no_grad():
            for firstPhotos, secondPhotos, monthsCount in test_loader:
                firstPhotos, secondPhotos, monthsCount = firstPhotos.to('cuda').float(), secondPhotos.to('cuda').float(), monthsCount.to('cuda').float()
                temp = torch.cat((firstPhotos, monthsCount.view(2, 1)), dim=1)
                if str(monthsCount.shape) == 'torch.Size([2])':
                    temp = torch.cat((firstPhotos, monthsCount.view(2, 1)), dim=1)
                else:
                    temp = torch.cat((firstPhotos.T, monthsCount.view(1, 1)))
                    temp = temp.T
                outputs = model(temp)
                total += secondPhotos.size(0)
                correct += (outputs.argmax(dim=1) == secondPhotos.argmax(dim=1)).sum().item()
        print(f'Test Accuracy: {100 * correct / total}%')
        torch.cuda.empty_cache()