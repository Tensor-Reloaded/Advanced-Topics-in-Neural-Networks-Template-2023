import torch
from torch import nn
from torch.utils.data import Subset
from typing import Callable
import wandb

def accuracy(output_list, label_list):
    false = torch.logical_not(output_list==label_list).sum().item()
    length = len(output_list)
    return (length-false)/length

def train(model: nn.Module, train_dataset: Subset, loss_fn: Callable, optimizer, epoch):
    model.train()
    total_loss = 0
    output_list = []
    label_list = []
    for batch, (data, labels) in enumerate(train_dataset):
        d = data
        l = labels
        label_list.append(l) 
        optimizer.zero_grad()
        outputs = model(d)
        output_list.append(outputs)
        loss = loss_fn(outputs, l)
        wandb.log({'Epoch': epoch, })
        loss.backward()
        optimizer.step()
        wandb.log({'Epoch': epoch, 'Batch': batch, 'Batch_loss:': loss.item()})
        total_loss += loss.item()
    
    train_accuracy = accuracy(torch.cat(output_list).argmax(dim=1), torch.cat(label_list))
    avg_loss = total_loss/len(train_dataset)
    wandb.log({'Train_accuracy': train_accuracy, 'Average_epoch_loss': avg_loss})
    return train_accuracy, avg_loss

def val(model: nn.Module, val_dataset: Subset, loss_fn: Callable, epoch):
    total_loss = 0
    output_list = []
    label_list = []
    for batch, (data, labels) in enumerate(val_dataset):
        d = data
        l = labels
        label_list.append(l)
        with torch.no_grad():
            outputs = model(d)
        output_list.append(outputs)
        loss = loss_fn(outputs, l)
        total_loss += loss.item()

    val_accuracy = accuracy(torch.cat(output_list).argmax(dim=1), torch.cat(label_list))
    avg_loss = total_loss/len(val_dataset)
    wandb.log({'Validation_accuracy': val_accuracy, 'Average_epoch_loss': avg_loss})
    return val_accuracy, avg_loss