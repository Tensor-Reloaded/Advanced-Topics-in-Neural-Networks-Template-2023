# TODO:Define types of each parameters for every function,as well as return type
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import *
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import datetime
import wandb
from typing import Union
from models import *
from sam import SAM

__all__ = ['TrainingPipeline']


# TODO:What about adding a different arhitecture,say we split the image in 4 corners and 1 piece and try
#  combining the results at the end?


# TODO:Would increasing the dropout layer as we learn more and more help the network generalize even better?

def disable_bn(model):
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm1d):
            module.eval()


def enable_bn(model):
    model.train()


def build_optimizer(model: torch.nn.Module, config) -> torch.optim:
    optimizer = None
    optimizer_name = config.optimizer_name
    if optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=config.lr, momentum=config.momentum,
                                    weight_decay=config.weight_decay, nesterov=config.nesterov)
    elif optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr,
                                     weight_decay=config.weight_decay)
    elif optimizer_name == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=config.lr, momentum=config.momentum,
                                        weight_decay=config.weight_decay)
    elif optimizer_name == 'Adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=config.lr,
                                        weight_decay=config.weight_decay, lr_decay=config.lr_decay)
    elif optimizer_name == 'SAM with SGD':
        base_optimizer = torch.optim.SGD
        optimizer = SAM(model.parameters(), base_optimizer,
                        lr=config.lr, momentum=config.momentum,
                        weight_decay=config.weight_decay, nesterov=config.nesterov)

    return optimizer


class TrainingPipeline:
    def __init__(self, device: torch.device, use_config_for_train: bool,
                 train_dataset: Dataset, val_dataset: Union[None, Dataset],
                 train_transformer, val_transformer,
                 cache: bool = True,
                 train_batch_size: Union[None, int] = 32, val_batch_size: Union[None, int] = 32,
                 no_workers: int = 2):
        self.device = device

        self.train_transformer = train_transformer
        self.val_transformer = val_transformer
        self.cache = cache
        self.no_workers = no_workers
        self.train_batch_size = train_batch_size

        self.train_loader = None
        self.validation_loader = None
        self.train_dataset = None
        self.val_dataset = None

        self.model = None

        if not use_config_for_train:
            self.build_data_loaders(train_dataset, val_dataset, train_transformer, val_transformer, cache,
                                    train_batch_size,
                                    val_batch_size, no_workers)
        else:
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset

    def build_data_loaders(self, train_dataset: Dataset, validation_dataset: Union[None, Dataset],
                           train_transformers, val_transformer,
                           cache: bool = True,
                           train_batch_size: int = 32, validation_batch_size: Union[None, int] = 32,
                           no_workers: int = 2,
                           build_val_loader: bool = True):
        train_dataset = CachedDataset(train_dataset, train_transformers, cache)

        pin_memory = (self.device.type == 'cuda')
        persistent_workers = (no_workers != 0)
        self.train_loader = DataLoader(train_dataset, shuffle=True, pin_memory=pin_memory, num_workers=no_workers,
                                       batch_size=train_batch_size, drop_last=True,
                                       persistent_workers=persistent_workers)
        if build_val_loader:
            validation_dataset = CachedDataset(validation_dataset, val_transformer, cache)
            self.validation_loader = DataLoader(validation_dataset, shuffle=False, pin_memory=True, num_workers=0,
                                                batch_size=validation_batch_size, drop_last=False)

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

            output = output.softmax(dim=1).detach().cpu().squeeze()
            labels = labels.cpu().squeeze()
            all_outputs.append(output)
            all_labels.append(labels)

        all_outputs = torch.cat(all_outputs).argmax(dim=1).to(self.device, non_blocking=True)
        all_labels = torch.cat(all_labels).to(self.device, non_blocking=True)

        fp_plus_fn = torch.logical_not(all_outputs == all_labels).sum().item()
        all_elements = all_outputs.shape[0]

        # TODO:Consider whether we should divide the loss by /no_instances

        return (all_elements - fp_plus_fn) / all_elements, total_loss

    def train(self, criterion, optimizer, pbar, current_batch, writer) -> tuple[int, float, float]:
        # Returns the current_batch,accuracy and loss
        # The last 2 are computed by a moving average,instead of computing them at the end of
        # the train epoch for efficiency
        self.model.train()

        total_loss = 0.0
        all_outputs = []
        all_labels = []

        for data, labels in self.train_loader:
            data = data.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            output = self.model(data)
            loss = None

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 4)

            if isinstance(optimizer, SAM):
                # first forward-backward step
                enable_bn(self.model)  # <- this is the important line
                loss = criterion(output, labels)
                loss.mean().backward()
                optimizer.first_step(zero_grad=True)

                # second forward-backward step
                disable_bn(self.model)  # <- this is the important line
                output = self.model(data)
                criterion(output, labels).mean().backward()
                optimizer.second_step(zero_grad=True)
            else:
                loss = criterion(output, labels)
                loss.backward()

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            pbar.update()

            writer.add_scalar("Train/Batch Loss", loss.item() / len(labels), current_batch)

            total_loss += loss

            all_outputs.append(output.softmax(dim=1).detach().cpu().squeeze())
            all_labels.append(labels.cpu().squeeze())

            current_batch += 1

        all_outputs = torch.cat(all_outputs).argmax(dim=1).to(self.device, non_blocking=True)
        all_labels = torch.cat(all_labels).to(self.device, non_blocking=True)

        no_instances = all_outputs.shape[0]
        fp_plus_fn = torch.logical_not(all_outputs == all_labels).sum().item()

        return current_batch, (no_instances - fp_plus_fn) / no_instances, total_loss

    def get_model_norm(self):
        norm = 0.0
        if self.model is not None:
            for param in self.model.parameters():
                norm += torch.norm(param)
        return norm

    def run(self, no_epochs: int, model: nn.Module, criterion: nn.modules.loss, optimizer):
        self.model = model

        batch_size = self.train_batch_size
        optimizer_name = None
        if isinstance(optimizer, SAM):
            optimizer_name = 'SAM with ' + str(type(optimizer.base_optimizer).__name__)
        else:
            optimizer_name = type(optimizer).__name__

        learning_rate = optimizer.param_groups[0]['lr']
        time = datetime.datetime.now().strftime("%d-%m-%Y %H-%M-%S")
        model_name = f"{time} {optimizer_name}, Batch Size={batch_size}, Lr={learning_rate}"
        writer = SummaryWriter(log_dir=f"runs/train/{model_name}")

        current_batch = 1
        for epoch in range(no_epochs):
            pbar = tqdm(total=len(self.train_loader), desc=f"Epoch {epoch} ", dynamic_ncols=True)

            current_batch, train_accuracy, train_loss = self.train(criterion, optimizer, pbar,
                                                                   current_batch, writer)

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

    # Unlike the train() method,this should only be used for sweep by wandb
    # Because of that,we simply train and only measure the loss and accuracy via moving average
    def train_epoch(self, criterion, optimizer) -> tuple[float, float]:
        # Returns the current_batch,accuracy and loss
        # The last 2 are computed by a moving average,instead of computing them at the end of
        # the train epoch for efficiency
        self.model.train()

        total_loss = 0.0
        all_outputs = []
        all_labels = []

        for data, labels in self.train_loader:
            data = data.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            output = self.model(data)
            loss = None

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 4)

            if isinstance(optimizer, SAM):
                # first forward-backward step
                enable_bn(self.model)  # <- this is the important line
                loss = criterion(output, labels)
                total_loss += loss.item()

                loss.mean().backward()
                optimizer.first_step(zero_grad=True)

                # second forward-backward step
                disable_bn(self.model)  # <- this is the important line
                output = self.model(data)
                criterion(output, labels).mean().backward()
                optimizer.second_step(zero_grad=True)
            else:
                loss = criterion(output, labels)
                total_loss += loss.item()
                loss.backward()

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            all_outputs.append(output.softmax(dim=1).detach().cpu().squeeze())
            all_labels.append(labels.cpu().squeeze())

        all_outputs = torch.cat(all_outputs).argmax(dim=1).to(self.device, non_blocking=True)
        all_labels = torch.cat(all_labels).to(self.device, non_blocking=True)

        no_instances = all_outputs.shape[0]
        fp_plus_fn = torch.logical_not(all_outputs == all_labels).sum().item()

        return (no_instances - fp_plus_fn) / no_instances, total_loss

    def run_config(self, config=None):
        with wandb.init(config=config):
            config = wandb.config

            # Build train loader
            self.build_data_loaders(self.train_dataset, self.val_dataset, self.train_transformer,
                                    self.val_transformer,
                                    self.cache,
                                    config.batch_size, 500,
                                    self.no_workers, True)

            self.model = MLP(self.device, **config.model)
            optimizer = build_optimizer(self.model, config)

            criterion = nn.CrossEntropyLoss()

            for epoch in range(1, config.epochs + 1):
                train_accuracy, train_loss = self.train_epoch(criterion, optimizer)

                validation_accuracy, validation_loss = self.accuracy_and_loss("val_loader", criterion)

                wandb.log({"train_accuracy": train_accuracy, "epoch": epoch})
                wandb.log({"train_loss": train_loss, "epoch": epoch})

                wandb.log({"validation_accuracy": validation_accuracy, "epoch": epoch})
                wandb.log({"validation_loss": validation_loss, "epoch": epoch})
