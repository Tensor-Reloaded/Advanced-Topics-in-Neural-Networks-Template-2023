import datetime
from multiprocessing import freeze_support

import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import wandb
import yaml
from sam import SAM


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
        # For multi-gpu workstations, PyTorch will use the first available GPU (cuda:0), unless specified otherwise
        # (cuda:1).
    if torch.backends.mps.is_available():
        return torch.device('mos')
    return torch.device('cpu')


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
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
        # x = self.fc1(x)
        # x = self.relu(x)
        # x = self.fc2(x)
        # return x


def accuracy(output, labels):
    fp_plus_fn = torch.logical_not(output == labels).sum().item()
    all_elements = len(output)
    return (all_elements - fp_plus_fn) / all_elements


def train(model, train_loader, criterion, optimizer, device):
    model.train()

    all_outputs = []
    all_labels = []

    epoch_loss = 0
    for data, labels in train_loader:
        data = data.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if optimizer.__class__.__name__ == "SAM":
            def closure():
                loss1 = criterion(model(data), labels)
                loss1.backward()
                return loss1
            output = model(data)
            loss = criterion(output, labels)
            epoch_loss += loss
            loss.backward()
            optimizer.step(closure)
            optimizer.zero_grad()
        else:
            output = model(data)
            loss = criterion(output, labels)
            epoch_loss += loss

            loss.backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        output = output.softmax(dim=1).detach().cpu().squeeze()
        labels = labels.cpu().squeeze()
        all_outputs.append(output)
        all_labels.append(labels)

    all_outputs = torch.cat(all_outputs).argmax(dim=1)
    all_labels = torch.cat(all_labels)

    return round(accuracy(all_outputs, all_labels), 4), epoch_loss / len(train_loader)


def val(model, val_loader, criterion, device):
    model.eval()

    all_outputs = []
    all_labels = []

    epoch_loss = 0
    for data, labels in val_loader:
        data = data.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.no_grad():
            output = model(data)

        loss = criterion(output, labels)
        epoch_loss += loss
        output = output.softmax(dim=1).detach().cpu().squeeze()
        labels = labels.cpu().squeeze()
        all_outputs.append(output)
        all_labels.append(labels)

    all_outputs = torch.cat(all_outputs).argmax(dim=1)
    all_labels = torch.cat(all_labels)

    return round(accuracy(all_outputs, all_labels), 4), epoch_loss / len(val_loader)


def do_epoch(model, train_loader, val_loader, criterion, optimizer, device):
    acc, loss = train(model, train_loader, criterion, optimizer, device)
    acc_val, loss_val = val(model, val_loader, criterion, device)
    # torch.cuda.empty_cache()
    return acc, acc_val, loss, loss_val


def get_model_norm(model):
    norm = 0.0
    for param in model.parameters():
        norm += torch.norm(param)
    return norm


def main(device=get_default_device()):
    transforms = [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((28, 28), antialias=True),
        v2.Grayscale(),
        # v2.RandomHorizontalFlip(),
        # v2.RandomVerticalFlip(),
        # v2.RandomRotation((-70, 70)),
        v2.AutoAugment(v2.AutoAugmentPolicy.CIFAR10),
        torch.flatten,
    ]

    data_path = '../data'
    train_dataset = CIFAR10(root=data_path, train=True, transform=v2.Compose(transforms), download=True)
    val_dataset = CIFAR10(root=data_path, train=False, transform=v2.Compose(transforms), download=True)
    train_dataset = CachedDataset(train_dataset)
    val_dataset = CachedDataset(val_dataset)

    # config = {
    #     "epochs": 30,
    #     "l_input": 784,
    #     "l_hidden": 100,
    #     "l_output": 10,
    #     "learning_rate": 0.01,
    #     "optimizer": "SGD",
    #     "loss": "CrossEntropy",
    #     "batch_size": 256,
    #     "val_batch_size": 500,
    #     "activation": "ReLU",
    #     "nesterov": True,
    #     "momentum": 0.9
    # }
    # optimizer_name = config["optimizer"]

    with open("./sweep_config.yaml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    # optimizer_name = config['parameters']['optimizer']['value']

    curr_time = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
    wandb.init(project="atnn-lab5", sync_tensorboard=True, name=f"{curr_time}_overnight", config=config)
    config = wandb.config

    model = MLP(config.l_input, config.l_hidden, config.l_output)
    model = model.to(device)
    if config.optimizer.startswith("SGD_SAM"):
        base_optimizer = torch.optim.SGD
        optimizer = SAM(model.parameters(), base_optimizer, lr=config.learning_rate, momentum=config.momentum, nesterov=config.nesterov)
    elif config.optimizer.startswith("SGD"):
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum, nesterov=config.nesterov)
    elif config.optimizer.startswith("Adam"):
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    elif config.optimizer.startswith("RMSProp"):
        optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate, momentum=config.momentum)
    elif config.optimizer.startswith("AdaGrad"):
        optimizer = torch.optim.Adagrad(model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    num_workers = 2
    persistent_workers = (num_workers != 0)
    pin_memory = device.type == 'cuda'
    train_loader = DataLoader(train_dataset, shuffle=True, pin_memory=pin_memory, num_workers=num_workers,
                              batch_size=config.batch_size, drop_last=True, persistent_workers=persistent_workers)
    val_loader = DataLoader(val_dataset, shuffle=False, pin_memory=pin_memory, num_workers=0, batch_size=config.val_batch_size,
                            drop_last=False)

    writer = SummaryWriter()
    tbar = tqdm(tuple(range(config.epochs)))
    writer.add_scalar("Learning rate", config.learning_rate)
    writer.add_scalar("Batch size", config.batch_size)
    writer.add_text("Optimizer", optimizer.__class__.__name__)

    for epoch in tbar:
        acc, acc_val, loss, loss_val = do_epoch(model, train_loader, val_loader, criterion, optimizer, device)
        tbar.set_postfix_str(f"Acc: {acc}, Acc_val: {acc_val}, Loss: {loss}, Loss_val: {loss_val}")
        writer.add_scalar("Train/Accuracy", acc, epoch)
        writer.add_scalar("Val/Accuracy", acc_val, epoch)
        writer.add_scalar("Train/Loss", loss, epoch)
        writer.add_scalar("Val/Loss", loss_val, epoch)
        writer.add_scalar("Model/Norm", get_model_norm(model), epoch)
        wandb.log(
            {
                "epoch": epoch,
                "acc_train": acc,
                "train_loss": loss,
                "acc_val": acc_val,
                "loss_val": loss_val,
            })
    wandb.finish()


if __name__ == '__main__':
    freeze_support()
    main()
