
from multiprocessing import freeze_support

import wandb
import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm
from PIL import Image

import gc
from functools import wraps
from time import time
from torch import nn

from early_stopping import EarlyStopping
from torch.utils.tensorboard import SummaryWriter

from inference import load_model
from model import ImageTransformCNN
from sam import SAM


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mos')
    return torch.device('cpu')


def train(model, train_loader, criterion, optimizer, device,  writer, nr_epoch, optimizer_name):
    model.train()

    loss_per_epoch = 0
    length = len(train_loader)
    for step, (data, labels) in enumerate(train_loader):
        data = data.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if optimizer_name == "SGD+SAM":
            def closure():
                loss = criterion(model(data), labels)
                loss.backward()
                return loss

            output = model(data)
            loss = criterion(output, labels)
            loss_per_epoch += loss.item()
            loss.backward()
            optimizer.step(closure)
            optimizer.zero_grad()
        else:
            output = model(data)
            loss = criterion(output, labels)
            writer.add_scalar("batch_train_loss", loss, nr_epoch * length + step)
            loss_per_epoch += loss.item()
            loss.backward()

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    return loss_per_epoch / len(train_loader)


def val(model, val_loader, device, criterion):
    model.eval()

    loss_per_epoch = 0
    for data, labels in val_loader:
        data = data.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.no_grad():
            output = model(data)
            loss = criterion(output, labels)
            loss_per_epoch += loss

    return loss_per_epoch / len(val_loader)


def do_epoch(model, train_loader, val_loader, criterion, optimizer, device, writer, nr_epoch, optimizer_name):
    loss_per_train_epoch = train(model, train_loader, criterion, optimizer, device, writer,  nr_epoch, optimizer_name)
    loss_per_val_epoch = val(model, val_loader, device, criterion)
    return loss_per_train_epoch, loss_per_val_epoch


def get_model_norm(model):
    norm = 0.0
    for param in model.parameters():
        norm += torch.norm(param)
    return norm


def create_components(config, device):
    train_dataset = CustomDataset(train=True, cache=True)
    val_dataset = CustomDataset(train=False, cache=False)

    model = ImageTransformCNN()
    model = model.to(device)
    if config["optimizer"] == "RMSProp":
        optimizer = torch.optim.RMSprop(model.parameters(), config["learning_rate"], weight_decay=config["decay"], momentum=config["momentum"])
    elif config["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), config["learning_rate"], weight_decay=config["decay"])
    elif config["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), config["learning_rate"], weight_decay=config["decay"], momentum=config["momentum"])
    elif config["optimizer"] == "AdaGrad":
        optimizer = torch.optim.Adagrad(model.parameters(), config["learning_rate"], weight_decay=config["decay"])
    elif config["optimizer"] == "SGD+SAM":
        base_optimizer = torch.optim.SGD
        optimizer = SAM(model.parameters(), base_optimizer, lr=config["learning_rate"], weight_decay=config["decay"], momentum=config["momentum"])
    else:
        optimizer = torch.optim.SGD(model.parameters(), config["learning_rate"])
    # criterion = torch.nn.MSELoss()
    criterion = torch.nn.SmoothL1Loss()
    num_workers = 2
    persistent_workers = (num_workers != 0)
    pin_memory = device.type == 'cuda'
    train_loader = DataLoader(train_dataset, shuffle=True, pin_memory=pin_memory, num_workers=num_workers,
                              batch_size=config["batch_size_test"], drop_last=False, persistent_workers=persistent_workers)
    val_loader = DataLoader(val_dataset, shuffle=False, pin_memory=True, num_workers=0, batch_size=config["batch_size_val"],
                            drop_last=False)

    return model, train_loader, val_loader, criterion, optimizer


def save_trained_model(model, path):
    torch.save(model.state_dict(), path)


def main(hyperparameters, device=get_default_device()):
    writer = SummaryWriter("runs/Homework7_CNN_threeConvLayers_withBatchNormAndPooling_SmoothLoss_" + hyperparameters["optimizer"])
    with wandb.init(project="Homework7_CNN_threeConvLayers_withBatchNormAndPooling_SmoothLoss_" + hyperparameters["optimizer"], config=hyperparameters):
        config = wandb.config
        model, train_loader, val_loader, criterion, optimizer = create_components(config, device)

        writer.add_scalar("learning_rate", config.learning_rate)
        writer.add_scalar("batch_size_test", config.batch_size_test)
        writer.add_scalar("batch_size_val", config.batch_size_val)
        writer.add_text("optimizer", config.optimizer)

        wandb.watch(model, criterion, log="all", log_freq=10)
        tbar = tqdm(tuple(range(config.epochs)))
        early_stopping = EarlyStopping(epochs_permission=3, loss_difference=config["early_stopping"])
        for epoch in tbar:
            loss_per_train_epoch, loss_per_val_epoch = do_epoch(model, train_loader, val_loader, criterion, optimizer,
                                                                device, writer, epoch, config["optimizer"])

            tbar.set_postfix_str(f"Loss_train: {loss_per_train_epoch}, Loss_val: {loss_per_val_epoch}")
            wandb.log({"epoch": epoch, "train_loss": loss_per_train_epoch,
                       "validation_loss": loss_per_val_epoch, })
            writer.add_scalar("epoch_train_loss", loss_per_train_epoch, epoch)
            writer.add_scalar("epoch_val_loss", loss_per_val_epoch, epoch)
            writer.add_scalar("model_norm", get_model_norm(model), epoch)
            if early_stopping.verify_stopping_criteria(loss_per_val_epoch):
                break
        save_trained_model(model, hyperparameters["path"])
        return model


def timed(fn: callable):
    @wraps(fn)
    def wrap(*args, **kwargs):
        gc.collect()
        start = time()
        fn(*args, **kwargs)
        end = time()
        return end - start

    return wrap


def get_cifar10_images(data_path: str, train: bool):
    initial_transforms = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    cifar_10_images = CIFAR10(root=data_path, train=train, transform=initial_transforms, download=True)
    return [image for image, label in cifar_10_images]


class CustomDataset(Dataset):
    def __init__(self, data_path: str = './data', train: bool = True, cache: bool = True):
        self.images = get_cifar10_images(data_path, train)
        self.cache = cache
        self.transforms = v2.Compose([
            v2.Resize((28, 28), antialias=True),
            v2.Grayscale(),
            v2.functional.hflip,
            v2.functional.vflip,
        ])
        if cache:
            self.labels = [self.transforms(x) for x in self.images]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        if self.cache:
            return self.images[i], self.labels[i]
        return self.images[i], self.transforms(self.images[i])


@timed
def transform_dataset_with_transforms(dataset: TensorDataset):
    transforms = v2.Compose([
        v2.Resize((28, 28), antialias=True),
        v2.Grayscale(),
        v2.functional.hflip,
        v2.functional.vflip,
    ])
    for image in dataset.tensors[0]:
        transforms(image)


@timed
@torch.no_grad()
def transform_dataset_with_model(dataset, model: nn.Module, batch_size: int, device, pin_memory:bool):
    model = model.to(device)
    model.eval()
    dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size, pin_memory=pin_memory)
    for images in dataloader:
        images = images[0].to(device, non_blocking=True)
        result = model(images)


def test_inference_time(model: nn.Module, device=torch.device('cpu')):
    test_dataset = CustomDataset(train=False, cache=False)

    batch_size = 300
    pin_memory = True

    test_dataset = torch.stack(test_dataset.images)
    test_dataset = TensorDataset(test_dataset)
    t2 = transform_dataset_with_model(test_dataset, model, batch_size, device, pin_memory)
    t1 = transform_dataset_with_transforms(test_dataset)
    print(f"Sequential transforming each image took: {t1} on CPU. \n"
          f"Using a model with batch_size: {batch_size} took {t2} on {device.type}. \n")


if __name__ == '__main__':
    configs = []

    # SGD_SAM1 = dict(
    #     epochs=100,
    #     batch_size_test=70,
    #     batch_size_val=500,
    #     learning_rate=0.0025,
    #     momentum=0.9,
    #     decay=0.0005,
    #     early_stopping = 0.002,
    #     dataset="CIFAR-10",
    #     path="trainedModels/CNN_twoConvLayers_MSE.pt",
    #     architecture="CNN",
    #     optimizer="SGD+SAM",
    # )

    # SGD_SAM1 = dict(
    #     epochs=35,
    #     batch_size_test=80,
    #     batch_size_val=500,
    #     learning_rate=0.0025,
    #     momentum=0.9,
    #     decay=0.0005,
    #     early_stopping = 0.002,
    #     dataset="CIFAR-10",
    #     path="trainedModels/CNN_threeConvLayers_withBatchNormAndPooling_MSE_35epochs.pt",
    #     architecture="CNN",
    #     optimizer="SGD+SAM",
    # )
    SGD_SAM1 = dict(
        epochs=50,
        batch_size_test=80,
        batch_size_val=500,
        learning_rate=0.0025,
        momentum=0.9,
        decay=0.0005,
        early_stopping=0.002,
        dataset="CIFAR-10",
        path="trainedModels/CNN_threeConvLayers_withBatchNormAndPooling_SmoothLoss_50epochs.pt",
        architecture="CNN",
        optimizer="SGD+SAM",
    )
    configs.append(SGD_SAM1)

    # Commented the training part in order to use the loaded model
    # freeze_support()
    # for config in configs:
    #     main(config)

    model = load_model("trainedModels/CNN_threeConvLayers_withBatchNormAndPooling_SmoothLoss_50epochs.pt")
    test_inference_time(model, device=get_default_device())
