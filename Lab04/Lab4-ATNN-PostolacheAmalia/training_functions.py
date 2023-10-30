from torch import nn
from torch.utils.data import Subset
from typing import Callable
from tqdm import tqdm


def run(model: nn.Module, train_dataset: Subset, val_dataset: Subset, \
    epochNumber: int, loss_fn: Callable, optimizer):
    train_loss_means = []
    val_loss_means = []

    for epoch in range(epochNumber):
        train_loss = train(model, train_dataset, loss_fn, optimizer)
        train_loss.means.append(train_loss)
        val_loss = val(model, val_dataset, loss_fn)
        val_loss_means.append(val_loss)
        print(f'Epoch {epoch+1}/{epochNumber}, training loss: {train_loss}, validation loss: {val_loss}\n')
    
    return train_loss_means, val_loss_means

def train(model: nn.Module, train_dataset: Subset, loss_fn: Callable, optimizer):
    model.train()
    total_loss = 0
    pbar = tqdm(total=len(train_dataset), desc="Training", dynamic_ncols=True)
    for start, end, time in train_dataset:
        optimizer.zero_grad()
        outputs = model((start, time))
        loss = loss_fn(outputs, end)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        pbar.set_postfix({'Loss': loss.item()})
        pbar.update()
    pbar.close()
    return total_loss / len(train_dataset)

def val(model: nn.Module, val_dataset: Subset, loss_fn: Callable):
    total_loss = 0
    for imageTuple in val_dataset:
        outputs = model((imageTuple[0], imageTuple[2]))
        loss = loss_fn(outputs, imageTuple[1])
        total_loss += loss.item()
    
    return total_loss / len(val_dataset)
