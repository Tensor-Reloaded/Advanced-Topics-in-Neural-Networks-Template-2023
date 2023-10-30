import torch
import torch.nn as nn
from typing import Callable
from torch.utils.data import Subset

class Pipeline:
    def __init__(self, model, optimizer, loss_function, train_loader, validation_loader, test_loader):
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function

        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader

    def train(self) -> float:
        self.model.train()
        total_loss = 0

        for in_img, out_img, time_skip in self.train_loader:
            self.optimizer.zero_grad()
            outputs = self.model(in_img, time_skip)
            
            loss = self.loss_function(outputs, out_img)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        print(total_loss, len(self.train_loader.dataset))
        return total_loss / len(self.train_loader.dataset)

    def val(self) -> float:
        total_loss = 0
        
        for in_img, out_img, time_skip in self.validation_loader:
            outputs = self.model(in_img, time_skip)
            total_loss += self.loss_function(outputs, out_img).item()

        return total_loss / len(self.validation_loader.dataset)
        
    def run(self, nr_epochs: int):

        train_loss_evolution = []
        validation_loss_evolution = []

        for epoch in range(nr_epochs):
            train_loss = self.train()
            val_loss = self.val()

            train_loss_evolution.append(train_loss)
            validation_loss_evolution.append(val_loss)

            print(f'Epoch {epoch + 1}/{nr_epochs}, Training Loss: {train_loss}\n')
            print(f'Epoch {epoch + 1}/{nr_epochs}, Validation Loss: {val_loss}\n')

        return train_loss_evolution, validation_loss_evolution





