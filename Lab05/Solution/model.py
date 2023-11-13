from logger import Logger
import time
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from typing import List, Optional, Type
from tqdm import tqdm

# TODO: normalization? See lab6 OptimizationRelatedMisc.ipynb

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
        self.layers.add_module('normalization_input', nn.BatchNorm1d(hidden_layers[0]))
        self.layers.add_module('activation_input', activation_fns[0])

        for i in range(len(hidden_layers) - 1):
            self.layers.add_module(f'hidden{i}', nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            self.layers.add_module(f'normalization_{i}', nn.BatchNorm1d(hidden_layers[i+1]))
            self.layers.add_module(f'activation{i}', activation_fns[i+1])
        
        self.layers.add_module('output', nn.Linear(hidden_layers[-1], output_size))

    def to(self, device):
        self.device = device
        return super().to(device)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

class Trainer:
    def __init__(
        self, 
        model: nn.Module, 
        criterion: nn.Module, 
        optimizer: optim.Optimizer, 
        logger: Optional[Logger], 
    ):
        self.__model__ = model
        self.__criterion__ = criterion
        self.__optimizer__ = optimizer
        self.__logger__ = logger

    def __train(self, dataloader: DataLoader, current_epoch: int) ->float:
        self.__model__.train()

        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        for batch_index, (inputs, targets) in enumerate(dataloader):

            inputs = inputs.to(self.__model__.device)
            targets = targets.to(self.__model__.device)

            outputs = self.__model__(inputs)

            def get_loss():
                self.__optimizer__.zero_grad()
                loss = self.__criterion__(outputs, targets)
                loss.backward()
                return loss
            
            loss = get_loss()

            self.__optimizer__.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += targets.size(0)
            correct_predictions += (predicted == targets).sum().item()

            self.__logger__.log_scalar_for_batch('Batch training loss', loss.item(), current_epoch * len(dataloader) + batch_index)

        accuracy = correct_predictions / total_predictions
        avg_loss = total_loss / len(dataloader)
        self.__logger__.log_scalar_for_epoch('Epoch training loss', avg_loss, current_epoch)
        self.__logger__.log_scalar_for_epoch('Epoch training accuracy', accuracy, current_epoch)

        return avg_loss, accuracy

    def __validate(self, dataloader: DataLoader, current_epoch: int) ->float:
        self.__model__.eval()

        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for inputs, targets in dataloader:

                inputs = inputs.to(self.__model__.device)
                targets = targets.to(self.__model__.device)

                outputs = self.__model__(inputs)
                loss = self.__criterion__(outputs, targets)
                total_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total_predictions += targets.size(0)
                correct_predictions += (predicted == targets).sum().item()

        accuracy = correct_predictions / total_predictions
        avg_loss = total_loss / len(dataloader)

        self.__logger__.log_scalar_for_epoch('Epoch validation loss', avg_loss, current_epoch)
        self.__logger__.log_scalar_for_epoch('Epoch validation accuracy', accuracy, current_epoch)

        return avg_loss, accuracy

    def run(self, train_dataloader: DataLoader, validation_dataloader: DataLoader, epochs: int):

        pbar = tqdm(range(epochs), desc="Training", unit="epoch")
        total_time = 0

        for epoch in pbar:
            start_time = time.time()

            train_loss, train_accuracy = self.__train(train_dataloader, epoch)
            validation_loss, validation_accuracy = self.__validate(validation_dataloader, epoch)

            model_norm = sum(p.norm().item() for p in self.__model__.parameters())
            self.__logger__.log_scalar_for_epoch('Model norm', model_norm, epoch)

            end_time = time.time()
            epoch_time = end_time - start_time
            total_time += epoch_time

            pbar.set_postfix({
                'Train Loss': train_loss,
                'Train Acc': train_accuracy,
                'Validation Loss': validation_loss,
                'Validation Acc': validation_accuracy,
                'Elapsed time (s)': total_time
            })