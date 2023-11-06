import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from typing import List, Type

from torch.utils.tensorboard import SummaryWriter

class Model(nn.Module):
    def __init__(self, input_size: int, hidden_layers: List[int], output_size: int, activation_fns: List[Type[nn.Module]]):
        super(Model, self).__init__()
        
        if not hidden_layers:
            raise ValueError("hidden_layers must not be empty")
        
        if any(layer <= 0 for layer in hidden_layers):
            raise ValueError("hidden_layers must contain positive values")
        
        if len(hidden_layers) != len(activation_fns):
            raise ValueError("The number of activation functions must match the number of hidden layers")
        
        self.layers = nn.Sequential()

        self.layers.add_module('input', nn.Linear(input_size, hidden_layers[0]))
        self.layers.add_module('activation_input', activation_fns[0])

        for i in range(len(hidden_layers) - 1):
            self.layers.add_module(f'hidden{i}', nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            self.layers.add_module(f'activation{i}', activation_fns[i+1])
        
        self.layers.add_module('output', nn.Linear(hidden_layers[-1], output_size))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

class Trainer:
    def __init__(self, model: nn.Module, criterion: nn.Module, optimizer: optim.Optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.writer = SummaryWriter()

    def __train(self, dataloader: DataLoader, current_epoch: int) ->float:
        self.model.train()

        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        for batch_index, (inputs, targets) in enumerate(dataloader):

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += targets.size(0)
            correct_predictions += (predicted == targets).sum().item()

            self.writer.add_scalar('Batch training loss', loss.item(), current_epoch * len(dataloader) + batch_index)

        accuracy = correct_predictions / total_predictions
        avg_loss = total_loss / len(dataloader)
        self.writer.add_scalar('Epoch training loss', avg_loss, current_epoch)
        self.writer.add_scalar('Epoch training accuracy', accuracy, current_epoch)

        return avg_loss, accuracy

    def __validate(self, dataloader: DataLoader, current_epoch: int) ->float:
        self.model.eval()

        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for inputs, targets in dataloader:

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total_predictions += targets.size(0)
                correct_predictions += (predicted == targets).sum().item()

        accuracy = correct_predictions / total_predictions
        avg_loss = total_loss / len(dataloader)

        self.writer.add_scalar('Epoch validation loss', avg_loss, current_epoch)
        self.writer.add_scalar('Epoch validation accuracy', accuracy, current_epoch)

        return avg_loss, accuracy

    def run(self, train_dataloader: DataLoader, val_dataloader: DataLoader, epochs: int):
        for epoch in range(epochs):
            train_loss, train_accuracy = self.__train(train_dataloader, epoch)
            val_loss, val_accuracy = self.__validate(val_dataloader, epoch)

            model_norm = sum(p.norm().item() for p in self.model.parameters())
            self.writer.add_scalar('Model norm', model_norm, epoch)
            self.writer.add_scalar('Learning rate', self.optimizer.param_groups[0]['lr'], epoch)
            self.writer.add_scalar('Batch size', train_dataloader.batch_size, epoch)
            self.writer.add_text('Optimizer', str(self.optimizer), epoch)

            print(f"Epoch {epoch+1}/{epochs} finished. Training loss: {train_loss}, Training accuracy: {train_accuracy}, Validation loss: {val_loss}, Validation accuracy: {val_accuracy}")
