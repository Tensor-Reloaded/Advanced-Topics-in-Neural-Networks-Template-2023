
from multiprocessing import freeze_support
import wandb
import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from tqdm import tqdm

from augmentations import CutOrMixUp
from cached_dataset import CachedDataset
from models import CifraMLP
from torch.utils.tensorboard import SummaryWriter
from sam import SAM
from torch.utils.data import default_collate


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mos')
    return torch.device('cpu')


def accuracy(output, labels):
    fp_plus_fn = torch.logical_not(output == labels).sum().item()
    all_elements = len(output)
    return (all_elements - fp_plus_fn) / all_elements


def train(model, train_loader, criterion, optimizer, device, cutmix_or_mixup, writer, nr_epoch, optimizer_name):

    model.train()

    all_outputs = []
    all_labels = []
    loss_per_epoch = 0
    length = len(train_loader)
    for step, (data, labels) in enumerate(train_loader):
        #data, labels = cutmix_or_mixup(data, labels) # !!!
        #data = data.reshape(len(data), -1) # !!!
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

        output = output.softmax(dim=1).detach().cpu().squeeze()
        labels = labels.cpu().squeeze()
        all_outputs.append(output)
        all_labels.append(labels)

    all_outputs = torch.cat(all_outputs).argmax(dim=1)
    all_labels = torch.cat(all_labels)

    return round(accuracy(all_outputs, all_labels), 4), loss_per_epoch / len(train_loader)


def val(model, val_loader, device, criterion):
    model.eval()

    all_outputs = []
    all_labels = []
    loss_per_epoch = 0
    for data, labels in val_loader:
        #data = data.reshape(len(data), -1) # !!!
        data = data.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.no_grad():
            output = model(data)
            loss = criterion(output, labels)
            loss_per_epoch += loss

        output = output.softmax(dim=1).detach().cpu().squeeze()
        labels = labels.cpu().squeeze()
        all_outputs.append(output)
        all_labels.append(labels)

    all_outputs = torch.cat(all_outputs).argmax(dim=1)
    all_labels = torch.cat(all_labels)

    return round(accuracy(all_outputs, all_labels), 4), loss_per_epoch / len(val_loader)


def do_epoch(model, train_loader, val_loader, criterion, optimizer, device, cutmix_or_mixup, writer, nr_epoch, optimizer_name):
    acc, loss_per_train_epoch = train(model, train_loader, criterion, optimizer, device, cutmix_or_mixup, writer,  nr_epoch, optimizer_name)
    acc_val, loss_per_val_epoch = val(model, val_loader, device, criterion)
    return acc, acc_val, loss_per_train_epoch, loss_per_val_epoch


def get_model_norm(model):
    norm = 0.0
    for param in model.parameters():
        norm += torch.norm(param)
    return norm


def create_components(config, device):

    transforms = [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((28, 28), antialias=True),
        v2.Grayscale(),
        torch.flatten,
    ]

    data_path = '../data'
    train_dataset = CIFAR10(root=data_path, train=True, transform=v2.Compose(transforms), download=True)
    val_dataset = CIFAR10(root=data_path, train=False, transform=v2.Compose(transforms), download=True)
    train_dataset = CachedDataset(train_dataset)
    val_dataset = CachedDataset(val_dataset)

    model = CifraMLP(config["input_dim"], 300, 150, config["classes"])
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
    criterion = torch.nn.CrossEntropyLoss()
    num_workers = 2
    persistent_workers = (num_workers != 0)
    pin_memory = device.type == 'cuda'
    train_loader = DataLoader(train_dataset, shuffle=True, pin_memory=pin_memory, num_workers=num_workers,
                              batch_size=config["batch_size_test"], drop_last=False, persistent_workers=persistent_workers)
    val_loader = DataLoader(val_dataset, shuffle=False, pin_memory=True, num_workers=0, batch_size=config["batch_size_val"],
                            drop_last=False)

    return model, train_loader, val_loader, criterion, optimizer


def main(hyperparameters, device=get_default_device()):
    cutmix_or_mixup = CutOrMixUp()
    writer = SummaryWriter("runs/Lab5_" + hyperparameters["optimizer"])
    with wandb.init(project="Lab5_" + hyperparameters["optimizer"], config=hyperparameters):
        config = wandb.config
        model, train_loader, val_loader, criterion, optimizer = create_components(config, device)

        writer.add_scalar("learning_rate", config.learning_rate)
        writer.add_scalar("batch_size_test", config.batch_size_test)
        writer.add_scalar("batch_size_val", config.batch_size_val)
        writer.add_text("optimizer", config.optimizer)

        wandb.watch(model, criterion, log="all", log_freq=10)
        tbar = tqdm(tuple(range(config.epochs)))
        for epoch in tbar:
            acc, acc_val, loss_per_train_epoch, loss_per_val_epoch = do_epoch(model, train_loader, val_loader,
                                                                              criterion, optimizer, device,
                                                                              cutmix_or_mixup, writer, epoch, config["optimizer"])

            tbar.set_postfix_str(f"Acc: {acc}, Acc_val: {acc_val}")
            wandb.log({"epoch": epoch, "train_loss": loss_per_train_epoch, "train_accuracy": acc,
                       "validation_loss": loss_per_val_epoch, "validation_accuracy": acc_val})
            writer.add_scalar("epoch_train_accuracy", acc, epoch)
            writer.add_scalar("epoch_train_loss", loss_per_train_epoch, epoch)
            writer.add_scalar("epoch_val_accuracy", acc_val, epoch)
            writer.add_scalar("epoch_val_loss", loss_per_val_epoch, epoch)
            writer.add_scalar("model_norm", get_model_norm(model), epoch)


if __name__ == '__main__':
    configs = []
    # Create configurations for the SGD optimizer
    SGD_conf1 = dict(
        epochs=50,
        classes=10,
        batch_size_test=100,
        batch_size_val=500,
        input_dim=784,
        learning_rate=0.01,
        momentum=0.0,
        decay=0.0,
        dataset="CIFAR-10",
        architecture="MLP",
        optimizer="SGD",
    )
    configs.append(SGD_conf1)

    SGD_conf2 = dict(
        epochs=75,
        classes=10,
        batch_size_test=125,
        batch_size_val=500,
        input_dim=784,
        learning_rate=0.02,
        momentum=0.9,
        decay=0.0005,
        dataset="CIFAR-10",
        architecture="MLP",
        optimizer="SGD",
    )
    configs.append(SGD_conf2)

    SGD_conf3 = dict(
        epochs=80,
        classes=10,
        batch_size_test=50,
        batch_size_val=500,
        input_dim=784,
        learning_rate=0.01,
        momentum=0.0,
        decay=0.0,
        dataset="CIFAR-10",
        architecture="MLP",
        optimizer="SGD",
    )
    configs.append(SGD_conf3)

    # Create copies of the created configurations and update them in order to use different optimizers
    Adam_conf1 = SGD_conf1.copy()
    Adam_conf1["optimizer"] = "Adam"
    configs.append(Adam_conf1)

    Adam_conf2 = SGD_conf2.copy()
    Adam_conf2["optimizer"] = "Adam"
    configs.append(Adam_conf2)

    Adam_conf3 = SGD_conf3.copy()
    Adam_conf3["optimizer"] = "Adam"
    configs.append(Adam_conf3)

    RMSProp_conf1 = SGD_conf1.copy()
    RMSProp_conf1["optimizer"] = "RMSProp"
    configs.append(RMSProp_conf1)

    RMSProp_conf2 = SGD_conf2.copy()
    RMSProp_conf2["optimizer"] = "RMSProp"
    configs.append(RMSProp_conf2)

    RMSProp_conf3 = SGD_conf3.copy()
    RMSProp_conf3["optimizer"] = "RMSProp"
    configs.append(RMSProp_conf3)

    AdaGrad_conf1 = SGD_conf1.copy()
    AdaGrad_conf1["optimizer"] = "AdaGrad"
    configs.append(AdaGrad_conf1)

    AdaGrad_conf2 = SGD_conf2.copy()
    AdaGrad_conf2["optimizer"] = "AdaGrad"
    AdaGrad_conf2["momentum"] = 0.9
    configs.append(AdaGrad_conf2)

    AdaGrad_conf3 = SGD_conf3.copy()
    AdaGrad_conf3["optimizer"] = "AdaGrad"
    configs.append(AdaGrad_conf3)

    SGD_SAM1 = dict(
        epochs=150,
        classes=10,
        batch_size_test=100,
        batch_size_val=500,
        input_dim=784,
        learning_rate=0.005,
        momentum=0.9,
        decay=0.0,
        dataset="CIFAR-10",
        architecture="MLP",
        optimizer="SGD+SAM",
    )
    configs.append(SGD_SAM1)

    SGD_SAM2 = SGD_SAM1.copy()
    SGD_SAM2["decay"] = 0.0005
    SGD_SAM2["learning_rate"] = 0.01
    configs.append(SGD_SAM2)

    SGD_SAM3 = SGD_SAM1.copy()
    SGD_SAM3["epochs"] = 85
    SGD_SAM3["learning_rate"] = 0.01
    SGD_SAM3["batch_size_test"] = 85
    configs.append(SGD_SAM3)

    freeze_support()
    for config in configs:
        main(config)
    