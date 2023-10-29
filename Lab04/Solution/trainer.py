from torch import nn
from torch.utils.data import Subset
from tqdm import tqdm
from typing import Callable


def run(nr_epochs: int, model: nn.Module, train_data: Subset, val_data: Subset, loss_fn: Callable, optimizer) \
        -> tuple[list, list]:

    train_loss_evo = []
    val_loss_evo = []

    for epoch in range(nr_epochs):
        train_loss = train(model, train_data, loss_fn, optimizer)
        val_loss = val(model, val_data, loss_fn)

        train_loss_evo.append(train_loss)
        val_loss_evo.append(val_loss)

        print(f'Epoch {epoch + 1}/{nr_epochs}, Training Loss: {train_loss}\n')
        print(f'Epoch {epoch + 1}/{nr_epochs}, Validation Loss: {val_loss}\n')

    return train_loss_evo, val_loss_evo


def train(model: nn.Module, train_data: Subset, loss_fn: Callable, optimizer) -> float:
    model.train()
    total_loss = 0
    pbar = tqdm(total=len(train_data), desc="Training", dynamic_ncols=True)

    for in_img, out_img, nr_days in train_data:
        optimizer.zero_grad()
        outputs = model((in_img, nr_days))
        loss = loss_fn(outputs, out_img)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        pbar.set_postfix({'Loss': loss.item()})
        pbar.update()

    pbar.close()
    return total_loss / len(train_data)


def val(model: nn.Module, val_data: Subset, loss_fn: Callable) -> float:
    total_loss = 0
    for in_img, out_img, nr_days in val_data:
        outputs = model((in_img, nr_days))
        total_loss += loss_fn(outputs, out_img).item()

    return total_loss / len(val_data)
