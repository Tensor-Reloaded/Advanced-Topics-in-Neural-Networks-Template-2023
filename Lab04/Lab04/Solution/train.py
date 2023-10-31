import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from Lab04.Lab04.Solution.models import ImagePredictionModel
from Lab04.Lab04.Solution.transform import RotateImage, BlurImage, CropImage
from Lab04.Lab04.Solution.dataset import ImageDataset

BATCH_SIZE = 100
SCALE_RATIO = 255


def train(epoch_index, model, val_loader, train_loader, optimizer):
    criterion = nn.MSELoss()

    model.train()
    total_loss = 0
    correct = 0

    pbar = tqdm(total=len(train_loader), desc="Training", dynamic_ncols=True)

    for features, labels, days in train_loader:
        if len(features) < BATCH_SIZE:
            pbar.update()
            continue

        optimizer.zero_grad()
        outputs = model(features, days)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_loss += labels.size(0)

        scaled_outputs = (outputs * SCALE_RATIO).to(torch.int)
        scaled_labels = (labels * SCALE_RATIO).to(torch.int)

        match_count = torch.sum(torch.eq(scaled_labels, scaled_outputs))
        correct += match_count.item()

        pbar.set_postfix({'Loss': loss.item(), 'Correct': match_count.item()})
        pbar.update()

    pbar.close()

    pbar = tqdm(total=len(val_loader), desc="Validation", dynamic_ncols=True)
    model.eval()
    total_values = 0
    validation_loss = 0
    match_count = 0

    with torch.no_grad():
        for inputs, labels, days in val_loader:
            outputs = model(inputs, days)
            validation_loss += criterion(outputs, labels)
            total_values += len(labels)

            scaled_outputs = (outputs * BATCH_SIZE).to(torch.int)
            scaled_labels = (labels * BATCH_SIZE).to(torch.int)

            elementwise_equal = torch.eq(scaled_labels, scaled_outputs)
            row_equal = elementwise_equal.all(dim=1)
            match_count += row_equal.sum().item()

            pbar.set_postfix({'Loss': validation_loss.item(), 'Accuracy': (match_count / total_values) * 100})
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
            scaled_outputs = (outputs * SCALE_RATIO).to(torch.int)
            scaled_labels = (labels * SCALE_RATIO).to(torch.int)

            elementwise_equal = torch.eq(scaled_labels, scaled_outputs)
            row_equal = elementwise_equal.all(dim=1)
            total += len(features)
            correct += row_equal.sum().item()

            loss = criterion(outputs, labels)
            total_loss += loss.item()

    accuracy = correct / total
    average_loss = total_loss / total
    print(f"Accuracy: {accuracy}, Loss: {average_loss}")


def run(n_epochs):
    total_dataset = ImageDataset(dataset_file="Homework Dataset", transform_list=[RotateImage()])
    print("Number of samples:", len(total_dataset))

    # Split the dataset into training, validation, and Solution sets...
    # train_dataset = int(0.7 * len(total_dataset))
    # validation_dataset = int(0.15 * len(total_dataset))
    # test_dataset = len(total_dataset) - train_dataset - validation_dataset
    #
    # train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # val_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(total_dataset, [0.7, 0.15, 0.15])

    model = ImagePredictionModel(input_size=128 * 128 * 3 + 1, hidden_size=1000, output_size=128 * 128 * 3,
                                 output_activation=nn.Sigmoid())

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(n_epochs):
        train(epoch, model, validation_dataset, train_dataset, optimizer)

    model_test(model, test_dataset)
