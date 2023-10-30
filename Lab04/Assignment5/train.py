from typing import List

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from Assignment5.model import Model
from Assignment5.plotter import MetricsMemory


class TrainTune:
    def __init__(self, cmodel: Model,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 optimizers: List = None,
                 optimizer_args: List[dict] = None,
                 default_optim_args: dict = None,
                 device: torch.device = torch.device('cpu')):
        if not default_optim_args:
            default_optim_args = {'lr': 0.01}
        self.model = cmodel
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizers = []
        self.device = device
        if optimizers and len(optimizers) > 0:
            for index, optimizer in enumerate(optimizers):
                if optimizer_args and len(optimizer_args) > index \
                        and optimizer_args[index] != {}:
                    self.optimizers.append(optimizer(self.model.parameters(),
                                                     **(optimizer_args[index])))
                else:
                    self.optimizers.append(optimizer(nn.ParameterList(self.model.parameters()),
                                                     **default_optim_args))

    def train(self):
        self.model.train()
        total_loss = 0
        correct = 0
        pbar = tqdm(total=len(self.train_loader), desc="Training", dynamic_ncols=True)
        for features, labels in self.train_loader:
            for index, optimizer in enumerate(self.optimizers):
                self.optimizers[index].zero_grad()
            outputs = self.model(features)
            labels = labels.to(self.device)
            correct += (not (outputs - labels).all())
            loss = self.model.loss(outputs, labels)
            loss.backward()
            for index, optimizer in enumerate(self.optimizers):
                self.optimizers[index].step()
            total_loss += loss.item()
            pbar.set_postfix({'Loss': loss.item()})
            pbar.update()
        pbar.close()
        return total_loss, correct

    def val(self):
        self.model.eval()
        correct = 0
        total_loss = 0
        with torch.no_grad():
            for features, labels in self.val_loader:
                labels = labels.to(self.device)
                outputs = self.model(features)
                correct += (not (outputs - labels).all())
                total_loss += self.model.loss(outputs, labels).item()
        return total_loss, correct

    def run(self, n: int):
        metrics_memory = MetricsMemory(n)
        for epoch in range(n):
            total_loss, correct_train = self.train()
            valid_loss, correct_test = self.val()
            train_metrics = (correct_train / len(self.train_loader),
                             total_loss / len(self.train_loader))
            val_metrics = (correct_test / len(self.val_loader),
                           valid_loss / len(self.val_loader))
            metrics_memory.update_metrics(epoch, val_metrics, train_metrics)
        metrics_memory.draw_plot()
