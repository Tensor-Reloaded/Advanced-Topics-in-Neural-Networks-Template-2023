import torch
import torch.nn as nn
import torch.optim as optim

from Lab04.Solution.models import ImagePredictionModel
from Lab04.Solution.transforms import RotateImageWrapper, ColorJitterWrapper, CropImageWrapper
from datasets import *
from torch.utils.data import DataLoader
from tqdm import tqdm

BATCH_SIZE = 100
SCALE_RATIO = 255


def train(epoch_index: int, model: ImagePredictionModel, val_loader: DataLoader, train_loader: DataLoader,
          optimizer: optim.Adam):
    criterion = nn.MSELoss()

    model.train()
    total_loss = 0
    pbar = tqdm(total=len(train_loader), desc="Training", dynamic_ncols=True)
    correct = 0
    for features, labels, days in train_loader:
        if len(features) < BATCH_SIZE:
            pbar.update()
            continue
        # features, labels = features.to('mps'), labels.to('mps')
        optimizer.zero_grad()
        outputs = model(features, days)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        total_loss += labels.size(0)

        scaled_outputs = (outputs * SCALE_RATIO).to(torch.int)  # Scale outputs and convert to integers
        scaled_labels = (labels * SCALE_RATIO).to(torch.int)

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

    pbar = tqdm(total=len(val_loader), desc="Validation", dynamic_ncols=True)
    model.eval()  # Set the model to evaluation mode
    total_values = 0
    validation_loss = 0
    with torch.no_grad():
        match_count = 0
        for inputs, labels, days in val_loader:  # Iterate over the validation data
            outputs = model(inputs, days)
            validation_loss += criterion(outputs, labels)
            total_values += len(labels)
            scaled_outputs = (outputs * BATCH_SIZE).to(torch.int)  # Scale outputs and convert to integers
            scaled_labels = (labels * BATCH_SIZE).to(torch.int)

            elementwise_equal = torch.eq(scaled_labels, scaled_outputs)

            # Check equality for each row
            row_equal = elementwise_equal.all(dim=1)

            # Count the number of matching rows
            match_count += row_equal.sum().item()

            pbar.set_postfix({'Loss': validation_loss, 'Accuracy': match_count / total_values * 100})
            pbar.update()


    pbar.close()
    print(f'Epoch {epoch_index} - Validation Loss: {validation_loss:.4f}')


def model_test(model, test_loader):
    criterion = nn.MSELoss()

    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    with torch.no_grad():
        for features, labels in test_loader:
            outputs = model(features)
            scaled_outputs = (outputs * SCALE_RATIO).to(torch.int)  # Scale outputs and convert to integers
            scaled_labels = (labels * SCALE_RATIO).to(torch.int)

            elementwise_equal = torch.eq(scaled_labels, scaled_outputs)

            # Check equality for each row
            row_equal = elementwise_equal.all(dim=1)
            total += len(features)
            # Count the number of matching rows
            correct += row_equal.sum().item()

            loss = criterion(outputs, labels)
            total_loss += loss.item()
    print(f"Accuracy: {correct / total}, Loss: {total_loss / total}")


def run(n: int):
    total_dataset = ImageDataset(dataset_file="./../Homework Dataset",
                                 transform_list=[CropImageWrapper()])

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
