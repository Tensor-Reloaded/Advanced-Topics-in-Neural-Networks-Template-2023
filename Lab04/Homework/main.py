import torch
import torch.nn as nn
import torch.optim as optim

from Lab04.Homework.models import ImagePredictionModel
from Lab04.Homework.transforms import RotateImageWrapper, ColorJitterWrapper
from datasets import *
from torch.utils.data import DataLoader
from tqdm import tqdm

BATCH_SIZE = 100


def train(epoch_index: int, model: ImagePredictionModel, val_loader: DataLoader, train_loader: DataLoader,
          optimizer: optim.Adam):
    criterion = nn.MSELoss()

    model.train()
    total_loss = 0
    pbar = tqdm(total=len(train_loader), desc="Training", dynamic_ncols=True)
    correct = 0
    for features, labels, days in train_loader:
        if len(features) < BATCH_SIZE:
            continue
        # features, labels = features.to('mps'), labels.to('mps')
        optimizer.zero_grad()
        outputs = model(features, days)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        total_loss += labels.size(0)

        scaled_outputs = (outputs * 1).to(torch.int)  # Scale outputs and convert to integers
        scaled_labels = (labels * 1).to(torch.int)

        match_count = 0
        for test_index in range(len(scaled_outputs)):
            if torch.equal(scaled_labels[test_index], scaled_outputs[test_index]):
                match_count += 1
        # matching_arrays = (scaled_outputs == scaled_labels).all(dim=1)
        # Count the number of matching arrays
        # match_count = matching_arrays.sum().item()
        correct += match_count

        pbar.set_postfix({'Loss': loss.item(), 'Correct': match_count})
        pbar.update()

    pbar.close()

    model.eval()  # Set the model to evaluation mode
    validation_loss = 0.0
    with torch.no_grad():
        match_count = 0
        for inputs, labels, days in val_loader:  # Iterate over the validation data
            outputs = model(inputs, days)
            validation_loss += criterion(outputs, labels)
            pbar.set_postfix({'Validation': validation_loss, 'Accuracy': match_count / len(val_loader) * 100})

            for test_index in range(len(outputs)):
                if torch.equal(outputs[test_index], labels[test_index]):
                    match_count += 1

        average_validation_loss = validation_loss / len(val_loader)

    print(f'Epoch - Validation Loss: {average_validation_loss:.4f}')

    # print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}\n')


def model_test(model, test_loader):
    criterion = nn.MSELoss()

    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    with torch.no_grad():
        for features, labels in test_loader:
            outputs = model(features)
            total += labels.size(0)
            correct += (outputs.argmax(dim=1) == labels.argmax(dim=1)).sum().item()

            loss = criterion(outputs, labels)
            total_loss += loss.item()


def run(n: int):
    total_dataset = ImageDataset(dataset_file="./../Homework Dataset", transform_list=[RotateImageWrapper(), ColorJitterWrapper()])

    print("Number of samples", len(total_dataset))
    train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(total_dataset, [0.7, 0.15, 0.15])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = ImagePredictionModel(input_size=128 * 128 * 3 + 1,
                                 hidden_size=1000,
                                 output_size=128 * 128 * 3,
                                 output_activation=nn.Sigmoid())

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(n):
        train(epoch, model, val_loader, train_loader, optimizer)

    model_test(model, test_loader)


if __name__ == '__main__':
    run(100)
