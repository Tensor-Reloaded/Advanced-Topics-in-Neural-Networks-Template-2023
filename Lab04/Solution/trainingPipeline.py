# TODO:Define types of each parameters for every function,as well as return type
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from models import *
from tqdm import tqdm

__all__ = ['TrainingPipeline']


class TrainingPipeline:
    def __init__(self, device: torch.device, dataset: Dataset,
                 datasets_proportions: tuple[float, float, float] = (0.7, 0.15, 0.15), batch_size: int = 32,
                 shuffle: bool = True):
        self.device = device

        train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(dataset, datasets_proportions)

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        self.validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=shuffle)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
        # TODO:Also save information about the hyper parameters given for the metrics

        self.model = None

        self.no_epochs = 0
        self.loss_training_per_epoch = []
        self.loss_validation_per_epoch = []

    def train(self, criterion, optimizer, pbar) -> float:
        self.model.train()
        total_loss = 0
        non_blocking = (self.device.type == 'cuda')
        for input_images, labels, months_between in self.train_loader:
            features = torch.cat(
                (input_images, torch.unsqueeze(months_between, 1).to(self.device, non_blocking=non_blocking)),
                dim=1).to(self.device, non_blocking=non_blocking)

            optimizer.zero_grad()
            outputs = self.model(features).to(self.device, non_blocking=non_blocking)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({'Loss': loss})
        return total_loss / len(self.train_loader)

    def val(self, criterion):
        self.model.eval()
        non_blocking = (self.device.type == 'cuda')
        with torch.no_grad():
            total_loss = 0
            for input_images, labels, months_between in self.train_loader:
                features = torch.cat(
                    (input_images, torch.unsqueeze(months_between, 1).to(self.device, non_blocking=non_blocking)),
                    dim=1)
                outputs = self.model(features)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def run(self, no_epochs: int, no_units_per_layer: list[int], output_activation):
        self.no_epochs = no_epochs
        self.model = MosaicMLP(self.device, no_units_per_layer, output_activation)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters())

        for epoch in range(no_epochs):
            pbar = tqdm(total=len(self.train_loader), desc="Training", dynamic_ncols=True)

            loss_training = self.train(criterion, optimizer, pbar)

            pbar.set_postfix({'Loss': loss_training})
            pbar.close()

            loss_validation = self.val(criterion)

            self.loss_training_per_epoch.append(loss_training)
            self.loss_validation_per_epoch.append(loss_validation)
