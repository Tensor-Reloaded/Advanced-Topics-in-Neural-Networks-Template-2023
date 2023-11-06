# TODO:Define types of each parameters for every function,as well as return type
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import *
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import datetime

__all__ = ['TrainingPipeline']


class TrainingPipeline:
    def __init__(self, device: torch.device, train_dataset: Dataset, validation_dataset: Dataset, train_transformers,
                 cache: bool = True,
                 train_batch_size: int = 32, validation_batch_size: int = 32, no_workers: int = 2):
        self.device = device
        self.train_batch_size = train_batch_size

        train_dataset = CachedDataset(train_dataset, train_transformers, cache)
        validation_dataset = CachedDataset(validation_dataset, None, cache)

        pin_memory = (device.type == 'cuda')
        persistent_workers = (no_workers != 0)
        self.train_loader = DataLoader(train_dataset, shuffle=True, pin_memory=pin_memory, num_workers=no_workers,
                                       batch_size=train_batch_size, drop_last=True,
                                       persistent_workers=persistent_workers)
        self.validation_loader = DataLoader(validation_dataset, shuffle=False, pin_memory=True, num_workers=0,
                                            batch_size=validation_batch_size, drop_last=False)

        self.model = None

        # self.no_epochs = 0
        # self.loss_training_per_epoch = []
        # self.loss_validation_per_epoch = []

    def accuracy_and_loss(self, loader_type: str, criterion) -> tuple[float, float]:
        self.model.eval()

        data_loader = self.validation_loader if loader_type == "val_loader" else self.train_loader

        all_outputs = []
        all_labels = []

        total_loss = 0.0

        for data, labels in data_loader:
            data = data.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with torch.no_grad():
                output = self.model(data)

            loss = criterion(output, labels)
            total_loss += loss.item()

            output = output.softmax(dim=1).cpu().squeeze()
            labels = labels.squeeze()
            all_outputs.append(output)
            all_labels.append(labels)

        all_outputs = torch.cat(all_outputs).argmax(dim=1).to(self.device, non_blocking=True)
        all_labels = torch.cat(all_labels).to(self.device, non_blocking=True)

        fp_plus_fn = torch.logical_not(all_outputs == all_labels).sum().item()
        all_elements = len(all_outputs)

        total_loss /= len(data_loader)

        return (all_elements - fp_plus_fn) / all_elements, total_loss

    def train(self, criterion, optimizer, pbar, current_batch, writer) -> int:
        self.model.train()

        for data, labels in self.train_loader:
            data = data.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            output = self.model(data)
            loss = criterion(output, labels)

            loss.backward()

            # TODO:How to use this together with Tensorboard in order to improve the model?
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            pbar.update()

            writer.add_scalar("Train/Batch Loss", loss.item() / len(labels), current_batch)

            current_batch += 1

        return current_batch

    def get_model_norm(self):
        norm = 0.0
        if self.model is not None:
            for param in self.model.parameters():
                norm += torch.norm(param)
        return norm

    def run(self, no_epochs: int, model: nn.Module, criterion, optimizer):
        self.model = model

        batch_size = self.train_batch_size
        optimizer_name = type(optimizer).__name__
        learning_rate = optimizer.param_groups[0]['lr']
        time = datetime.datetime.now().strftime("%d-%m-%Y %H-%M-%S")
        model_name = f"{time} {optimizer_name}, Batch Size={batch_size}, Lr={learning_rate}"
        writer = SummaryWriter(log_dir=f"runs/train/{model_name}")

        current_batch = 1
        for epoch in range(no_epochs):
            pbar = tqdm(total=len(self.train_loader), desc=f"Epoch {epoch} ", dynamic_ncols=True)

            current_batch = self.train(criterion, optimizer, pbar, current_batch, writer)

            train_accuracy, train_loss = self.accuracy_and_loss("train_loader", criterion)
            validation_accuracy, validation_loss = self.accuracy_and_loss("val_loader", criterion)

            pbar.set_postfix(
                {'Train Accuracy': train_accuracy,
                 'Validation Accuracy': validation_accuracy})

            pbar.close()

            writer.add_scalar("Train/Loss", train_loss, epoch + 1)
            writer.add_scalar("Train/Accuracy", train_accuracy, epoch + 1)
            writer.add_scalar("Val/Loss", validation_loss, epoch + 1)
            writer.add_scalar("Val/Accuracy", validation_accuracy, epoch + 1)
            writer.add_scalar("Model/Norm", self.get_model_norm(), epoch + 1)

        writer.close()
