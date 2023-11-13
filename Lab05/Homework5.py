import os
from multiprocessing import freeze_support
from typing import Optional

import torch
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Normalize
from torch.optim.lr_scheduler import CyclicLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from pydantic import BaseModel
import wandb

os.environ["WANDB_NOTEBOOK_NAME"] = "Homework5.ipynb"


class ModelConfiguration(BaseModel):
    epochs: int
    dropout_rate: Optional[float]
    batch_size_train: int
    batch_size_val: int
    layer_sizes: list[int]
    learning_rate: float
    momentum: Optional[float]
    weight_decay: Optional[float]
    base_lr: Optional[float]
    max_lr: Optional[float]
    step_size_up: Optional[int]
    optimizer: type
    optimizer_params: dict
    architecture: Optional[str] = "MLP"
    dataset: Optional[str] = "CIFAR-10"


def get_default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mos")
    return torch.device("cpu")


class CachedDataset(Dataset):
    def __init__(self, dataset, cache=True):
        if cache:
            dataset = tuple([x for x in dataset])
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return self.dataset[i]


class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = torch.nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc4 = torch.nn.Linear(hidden_sizes[1], output_size)
        self.relu = torch.nn.ReLU(inplace=True)
        self.sigmoid = torch.nn.Sigmoid()
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x):
        # return self.fc2(self.relu(self.fc1(x)))
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        # x = self.dropout(x)
        x = self.fc4(x)
        return x


def accuracy(output, labels):
    fp_plus_fn = torch.logical_not(output == labels).sum().item()
    all_elements = len(output)
    return (all_elements - fp_plus_fn) / all_elements


def train(
    model: torch.nn.Module,
    train_loader,
    criterion,
    optimizer: torch.optim.SGD,
    device,
    writer,
    epoch,
    learning_rate,
    batchSize,
    scheduler,
):
    model.train()

    all_outputs = []
    all_labels = []
    total_loss = 0.0
    index = 0

    for data, labels in train_loader:
        index += 1
        data = data.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        output = model(data)
        loss = criterion(output, labels)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        # optimizer.zero_grad(set_to_none=True)
        total_loss += loss.item()
        output = F.softmax(output, dim=1).cpu().squeeze()
        labels = labels.cpu().squeeze()
        all_outputs.append(output)
        all_labels.append(labels)

        writer.add_scalar(
            "Batch/Training_Loss", loss.item(), epoch * len(train_loader) + index
        )
        batch_acc = accuracy(output.argmax(dim=1), labels)
        writer.add_scalar(
            "Batch/Training_Accuracy", batch_acc, epoch * len(train_loader) + index
        )
        optimizer.zero_grad()
        scheduler.step()

    all_outputs = torch.cat(all_outputs).argmax(dim=1)
    all_labels = torch.cat(all_labels)

    epoch_acc = accuracy(all_outputs, all_labels)
    writer.add_scalar("Epoch/Training_Accuracy", epoch_acc, epoch)
    epoch_loss = total_loss / len(train_loader)
    writer.add_scalar("Epoch/Training_Loss", epoch_loss, epoch)

    writer.add_scalar("Train/Learning_Rate", learning_rate, epoch)
    writer.add_scalar("Train/Batch_Size", batchSize, epoch)
    writer.add_text("Train/Optimizer", str(optimizer), epoch)

    return round(accuracy(all_outputs, all_labels), 4)


def val(
    model,
    val_loader,
    criterion,
    optimizer,
    device,
    writer,
    epoch,
    learning_rate,
    batchSize,
):
    model.eval()

    all_outputs = []
    all_labels = []
    total_loss = 0.0
    index = 0

    with torch.no_grad():
        for data, labels in val_loader:
            index += 1
            data = data.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            output = model(data)
            loss = criterion(output, labels)
            total_loss += loss.item()
            output = F.softmax(output, dim=1).cpu().squeeze()
            labels = labels.cpu().squeeze()

            all_outputs.append(output)
            all_labels.append(labels)

            writer.add_scalar(
                "Batch/Val_Loss", loss.item(), epoch * len(val_loader) + index
            )
            batch_acc = accuracy(output.argmax(dim=1), labels)
            writer.add_scalar(
                "Batch/Val_Accuracy", batch_acc, epoch * len(val_loader) + index
            )

    all_outputs = torch.cat(all_outputs).argmax(dim=1)
    all_labels = torch.cat(all_labels)

    epoch_acc = accuracy(all_outputs, all_labels)
    writer.add_scalar("Epoch/Val_Accuracy", epoch_acc, epoch)
    epoch_loss = total_loss / len(val_loader)
    writer.add_scalar("Epoch/Val_Loss", epoch_loss, epoch)

    writer.add_scalar("Val/Learning_Rate", learning_rate, epoch)
    writer.add_scalar("Val/Batch_Size", batchSize, epoch)
    writer.add_text("Val/Optimizer", str(optimizer), epoch)

    return round(accuracy(all_outputs, all_labels), 4)


def do_epoch(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    writer,
    epoch,
    learning_rate,
    batchSize,
    scheduler,
):
    acc = train(
        model,
        train_loader,
        criterion,
        optimizer,
        device,
        writer,
        epoch,
        learning_rate,
        batchSize,
        scheduler,
    )
    acc_val = val(
        model,
        val_loader,
        criterion,
        optimizer,
        device,
        writer,
        epoch,
        learning_rate,
        batchSize,
    )
    # torch.cuda.empty_cache()
    return acc, acc_val


def get_model_norm(model):
    norm = 0.0
    for param in model.parameters():
        norm += torch.norm(param)
    return norm


def main(config: ModelConfiguration, device=get_default_device()):
    transforms = [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((28, 28), antialias=True),
        # v2.RandomRotation(degrees=15),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        v2.Grayscale(),
        # v2.RandomVerticalFlip(),
        # Normalize((0.5), (0.5)),
        torch.flatten,
    ]

    data_path = "../data"
    train_dataset = CIFAR10(
        root=data_path, train=True, transform=v2.Compose(transforms), download=False
    )
    val_dataset = CIFAR10(
        root=data_path, train=False, transform=v2.Compose(transforms), download=False
    )
    train_dataset = CachedDataset(train_dataset)
    val_dataset = CachedDataset(val_dataset)

    model = MLP(784, config.layer_sizes, 10, config.dropout_rate)
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()

    num_workers = 2

    persistent_workers = num_workers != 0
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        pin_memory=pin_memory,
        num_workers=num_workers,
        batch_size=config.batch_size_train,
        drop_last=True,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
        batch_size=config.batch_size_val,
        drop_last=False,
    )

    # torch.optim.s

    writer = SummaryWriter(
        comment=f"; {config.optimizer.__name__}; weight_decay={config.weight_decay}; base_lr={config.base_lr}; max_lr={config.max_lr}; step_size_up={config.step_size_up}; batch_size={config.batch_size_train}; dropout_rate={config.dropout_rate}; hidden_sizes={config.layer_sizes}"
    )

    tbar = tqdm(tuple(range(config.epochs)))

    # optimizer = torch.optim.SGD(
    #     model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    # )

    optimizer = config.optimizer(model.parameters(), **config.optimizer_params)

    scheduler = CyclicLR(
        optimizer,
        base_lr=config.base_lr,
        max_lr=config.max_lr,
        step_size_up=config.step_size_up,
        cycle_momentum=config.optimizer.__name__ not in ["Adam", "Adagrad"],
    )
    # scheduler = CosineAnnealingLR(optimizer, T_max=50)

    with wandb.init(
        project=f"Homework5_{config.optimizer.__name__}",
        config=config.model_dump(exclude={"optimizer"}),
    ):
        wandb.watch(model, criterion, log="all", log_freq=10)

        for epoch in tbar:
            # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0005)
            # optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=0.0005)
            # optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01, weight_decay=0.0001)

            acc, acc_val = do_epoch(
                model,
                train_loader,
                val_loader,
                criterion,
                optimizer,
                device,
                writer,
                epoch,
                config.learning_rate,
                config.batch_size_train,
                scheduler,
            )

            wandb.log(
                {"epoch": epoch, "train_accuracy": acc, "validation_accuracy": acc_val}
            )

            tbar.set_postfix_str(f"Acc: {acc}, Acc_val: {acc_val}")
            writer.add_scalar("Train/Accuracy", acc, epoch)
            writer.add_scalar("Val/Accuracy", acc_val, epoch)
            writer.add_scalar("Model/Norm", get_model_norm(model), epoch)


if __name__ == "__main__":
    freeze_support()

    configs = [
        ModelConfiguration(
            epochs=100,
            dropout_rate=0.3,
            batch_size_train=512,
            batch_size_val=512,
            layer_sizes=[512, 256],
            learning_rate=0.1,
            weight_decay=0.001,
            momentum=0.9,
            base_lr=0.04,
            max_lr=0.06,
            step_size_up=100,
            optimizer=torch.optim.SGD,
            optimizer_params={"lr": 0.1, "weight_decay": 0.001},
        ),
        ModelConfiguration(
            epochs=100,
            dropout_rate=0.3,
            batch_size_train=512,
            batch_size_val=512,
            layer_sizes=[512, 256],
            learning_rate=0.001,
            weight_decay=0.0001,
            momentum=0.9,
            base_lr=0.0005,
            max_lr=0.001,
            step_size_up=100,
            optimizer=torch.optim.Adam,
            optimizer_params={"lr": 0.001, "weight_decay": 0.0001},
        ),
        ModelConfiguration(
            epochs=100,
            dropout_rate=0.3,
            batch_size_train=512,
            batch_size_val=512,
            layer_sizes=[512, 256],
            learning_rate=0.001,
            weight_decay=0.0001,
            momentum=0.9,
            base_lr=1e-3,
            max_lr=1e-2,
            step_size_up=100,
            optimizer=torch.optim.Adagrad,
            optimizer_params={"lr": 1e-2, "weight_decay": 0.0001},
        ),
        ModelConfiguration(
            epochs=100,
            dropout_rate=0.3,
            batch_size_train=512,
            batch_size_val=512,
            layer_sizes=[600, 300],
            learning_rate=0.001,
            weight_decay=0.0001,
            momentum=0.9,
            base_lr=1e-5,
            max_lr=1e-4,
            step_size_up=50,
            optimizer=torch.optim.RMSprop,
            optimizer_params={"lr": 1e-4, "weight_decay": 0.0001},
        ),
        
        ModelConfiguration(
            epochs=100,
            dropout_rate=0.2,
            batch_size_train=512,
            batch_size_val=512,
            layer_sizes=[512, 256],
            learning_rate=0.1,
            weight_decay=0.001,
            momentum=0.9,
            base_lr=0.04,
            max_lr=0.06,
            step_size_up=50,
            optimizer=torch.optim.SGD,
            optimizer_params={"lr": 0.04, "weight_decay": 0.001},
        ),
        ModelConfiguration(
            epochs=100,
            dropout_rate=0.3,
            batch_size_train=512,
            batch_size_val=512,
            layer_sizes=[512, 256],
            learning_rate=0.001,
            weight_decay=0.0001,
            momentum=0.9,
            base_lr=1e-5,
            max_lr=1e-4,
            step_size_up=30,
            optimizer=torch.optim.Adam,
            optimizer_params={"lr": 1e-4, "weight_decay": 0.0001},
        ),
        ModelConfiguration(
            epochs=100,
            dropout_rate=0.3,
            batch_size_train=512,
            batch_size_val=512,
            layer_sizes=[512, 256],
            learning_rate=0.001,
            weight_decay=0.0001,
            momentum=0.9,
            base_lr=1e-4,
            max_lr=1e-3,
            step_size_up=50,
            optimizer=torch.optim.Adagrad,
            optimizer_params={"lr": 1e-3, "weight_decay": 0.0001},
        ),
        ModelConfiguration(
            epochs=100,
            dropout_rate=0.3,
            batch_size_train=512,
            batch_size_val=512,
            layer_sizes=[640, 480],
            learning_rate=0.001,
            weight_decay=0.0001,
            momentum=0.9,
            base_lr=1e-5,
            max_lr=1e-4,
            step_size_up=50,
            optimizer=torch.optim.RMSprop,
            optimizer_params={"lr": 1e-4, "weight_decay": 0.0001},
        ),



        ModelConfiguration(
            epochs=100,
            dropout_rate=0.1,
            batch_size_train=1024,
            batch_size_val=512,
            layer_sizes=[256, 128],
            learning_rate=0.1,
            weight_decay=0.001,
            momentum=0.9,
            base_lr=0.01,
            max_lr=0.1,
            step_size_up=50,
            optimizer=torch.optim.SGD,
            optimizer_params={"lr": 0.1, "weight_decay": 0.001},
        ),
        ModelConfiguration(
            epochs=100,
            dropout_rate=0.35,
            batch_size_train=1024,
            batch_size_val=512,
            layer_sizes=[512, 256],
            learning_rate=0.001,
            weight_decay=0.0001,
            momentum=0.9,
            base_lr=1e-5,
            max_lr=1e-4,
            step_size_up=30,
            optimizer=torch.optim.Adam,
            optimizer_params={"lr": 1e-4, "weight_decay": 0.001},
        ),
        ModelConfiguration(
            epochs=100,
            dropout_rate=0.35,
            batch_size_train=1024,
            batch_size_val=512,
            layer_sizes=[512, 256],
            learning_rate=0.001,
            weight_decay=0.0001,
            momentum=0.9,
            base_lr=1e-4,
            max_lr=1e-3,
            step_size_up=50,
            optimizer=torch.optim.Adagrad,
            optimizer_params={"lr": 1e-3, "weight_decay": 0.001},
        ),
        ModelConfiguration(
            epochs=100,
            dropout_rate=0.4,
            batch_size_train=512,
            batch_size_val=512,
            layer_sizes=[640, 480],
            learning_rate=0.001,
            weight_decay=0.0001,
            momentum=0.9,
            base_lr=1e-6,
            max_lr=1e-5,
            step_size_up=50,
            optimizer=torch.optim.RMSprop,
            optimizer_params={"lr": 1e-5, "weight_decay": 0.0001},
        ),
    ]
    for config in configs[8:]:
        main(config)
