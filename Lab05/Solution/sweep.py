import os
from multiprocessing import freeze_support
import wandb
import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
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
    def __init__(self, dataset, transforms=None, cache=True):
        if cache:
            dataset = tuple([x for x in dataset])
        self.dataset = dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        x = self.dataset[i][0]
        if self.transforms:
            x = self.transforms(x)
        y = torch.zeros(10)
        y[self.dataset[i][1]] = 1
        return x, y


class MLP(torch.nn.Module):
    def __init__(self, input_size, output_size, hid_dim, dropout):
        super(MLP, self).__init__()
        hid_dim = input_size
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_size, 4 * hid_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(4 * hid_dim, 8 * hid_dim),
            # torch.nn.ReLU(),
            # torch.nn.Dropout(p=dropout),
            # torch.nn.Linear(4*hid_dim, 4*hid_dim),
            # torch.nn.ReLU(),
            # torch.nn.Dropout(p=dropout),
            # torch.nn.Linear(4*hid_dim, 4*hid_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(8 * hid_dim, 2 * hid_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(2 * hid_dim, hid_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(hid_dim, output_size),

        )

    def forward(self, x):
        # return self.fc2(self.relu(self.fc1(x)))
        x = self.network(x)

        return x


def accuracy(output, labels):
    fp_plus_fn = torch.logical_not(output == labels).sum().item()
    all_elements = len(output)
    return (all_elements - fp_plus_fn) / all_elements


def train(model, train_loader, criterion, optimizer, scheduler, writer, epoch, using_sam, device):
    model.train()

    all_outputs = []
    all_labels = []
    total_loss = 0
    ite = 0
    for data, labels in train_loader:
        data = data.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        output = model(data)
        if not using_sam:
            # print(labels.shape)
            # print(output.shape)
            optimizer.zero_grad()
            loss = criterion(output, labels)
            # print(loss)
            ite += 1
            loss.backward()
            total_loss += loss.item()
            if writer:
                writer.add_scalar("Train/Batch_Loss", loss.item(), epoch * len(train_loader) + ite)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

            optimizer.step()
            scheduler.step()
        else:
            loss1 = criterion(output, labels)
            ite += 1
            loss1.backward()
            optimizer.first_step(zero_grad=True)
            output = model(data)
            loss2 = criterion(output, labels)
            loss2.backward()
            optimizer.second_step(zero_grad=True)
            loss = (loss1.item() + loss2.item()) / 2
            total_loss += loss
            if writer:
                writer.add_scalar("Train/Batch_Loss", loss, epoch * len(train_loader) + ite)

        output = output.softmax(dim=1).detach().cpu().squeeze()
        # print(output.shape)
        labels = labels.cpu().squeeze()
        all_outputs.append(output)
        all_labels.append(labels)
    total_loss /= len(train_loader)
    all_outputs = torch.cat(all_outputs).argmax(dim=1)
    all_labels = torch.cat(all_labels).argmax(dim=1)

    return round(accuracy(all_outputs, all_labels), 4), total_loss


def val(model, val_loader, criterion, device):
    model.eval()

    all_outputs = []
    all_labels = []
    total_loss = 0
    for data, labels in val_loader:
        data = data.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.no_grad():
            output = model(data)
        loss = criterion(output, labels)
        total_loss += loss.item()
        output = output.softmax(dim=1).cpu().squeeze()
        labels = labels.cpu().squeeze()  # .to(torch.bool)
        all_outputs.append(output)
        all_labels.append(labels)
    total_loss /= len(val_loader)
    all_outputs = torch.cat(all_outputs).argmax(dim=1)
    all_labels = torch.cat(all_labels).argmax(dim=1)  # .to(torch.bool)
    # print(all_labels)

    return round(accuracy(all_outputs, all_labels), 4), total_loss


def do_epoch(model, train_loader, val_loader, criterion, optimizer, scheduler, writer, epoch, using_sam, device):
    acc, train_loss = train(model, train_loader, criterion, optimizer, scheduler, writer, epoch, using_sam, device)
    acc_val, val_loss = val(model, val_loader, criterion, device)
    # torch.cuda.empty_cache()
    return acc, acc_val, train_loss, val_loss


def get_model_norm(model):
    norm = 0.0
    for param in model.parameters():
        norm += torch.norm(param)
    return norm


def start(device=get_default_device()):
    main_transforms = [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.49139968, 0.48215827, 0.44653124], std=[0.24703233, 0.24348505, 0.26158768]),
        v2.Resize((28, 28), antialias=True),
        v2.Grayscale(),

    ]
    train_transforms = [
        # v2.AutoAugment(v2.AutoAugmentPolicy.CIFAR10),
        # v2.RandomAffine(degrees=10, translate=(0.0, 0.1), scale=(0.8, 1.0)),
        v2.RandomHorizontalFlip(0.5),
        v2.RandomVerticalFlip(0.5),
        v2.RandomRotation(degrees=(-30, 30)),
        v2.ColorJitter(brightness=.3, contrast=0.3, saturation=0.3, hue=.3),

        torch.flatten,
    ]
    valid_transforms = [
        torch.flatten,
    ]
    # train_transforms = [
    # v2.RandomAffine(degrees=10, translate=(0.0, 0.1), scale=(0.8, 1.0)),
    # v2.RandomHorizontalFlip(0.25),
    # v2.RandomVerticalFlip(0.25),
    # v2.Normalize(mean=[0.49139968, 0.48215827 ,0.44653124], std=[0.24703233, 0.24348505, 0.26158768]),
    # v2.Resize((28, 28), antialias=True),
    # v2.Grayscale(),
    # torch.flatten,
    # ]
    # valid_transforms = [
    #     v2.Normalize(mean=[0.49139968, 0.48215827 ,0.44653124], std=[0.24703233, 0.24348505, 0.26158768]),
    #     v2.Resize((28, 28), antialias=True),
    #     v2.Grayscale(),
    #     torch.flatten,
    # ]

    data_path = '../data'
    train_dataset = CIFAR10(root=data_path, train=True, transform=v2.Compose(main_transforms), download=True)
    val_dataset = CIFAR10(root=data_path, train=False, transform=v2.Compose(main_transforms), download=True)
    train_dataset = CachedDataset(train_dataset, transforms=v2.Compose(train_transforms))
    val_dataset = CachedDataset(val_dataset, transforms=v2.Compose(valid_transforms))
    dropout = 0.4
    model = build_model(dropout, 850)
    model = model.to(device)
    learning_rate = 5e-3
    using_sam = True
    optimizer = build_optimizer(model, 'sam', learning_rate, 0.005)
    criterion = torch.nn.CrossEntropyLoss()
    epochs = 300
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)

    batch_size = 256
    val_batch_size = 500
    num_workers = 2
    print(device)
    persistent_workers = (num_workers != 0)
    pin_memory = device.type == 'cuda'
    train_loader = DataLoader(train_dataset, shuffle=True, pin_memory=pin_memory, num_workers=num_workers,
                              batch_size=batch_size, drop_last=True, persistent_workers=persistent_workers)
    val_loader = DataLoader(val_dataset, shuffle=False, pin_memory=True, num_workers=0, batch_size=val_batch_size,
                            drop_last=False)
    max_val = 0
    writer = SummaryWriter()
    tbar = tqdm(tuple(range(epochs)))

    for epoch in tbar:
        acc, acc_val, train_loss, val_loss = do_epoch(model, train_loader, val_loader, criterion, optimizer, scheduler,
                                                      writer, epoch, using_sam, device)
        if acc_val > max_val:
            max_val = acc_val
        tbar.set_postfix_str(f"Acc: {acc}, Acc_val: {acc_val}")
        # print("gata");
        writer.add_scalar("Train/Accuracy", acc, epoch)
        writer.add_scalar("Val/Accuracy", acc_val, epoch)
        writer.add_scalar("Train/Loss", train_loss, epoch)
        writer.add_scalar("Val/Loss", val_loss, epoch)
        writer.add_scalar("Model/Norm", get_model_norm(model), epoch)
    writer.add_hparams(
        {'lr': learning_rate, 'batch_size': batch_size, 'optimizer': optimizer.__class__.__name__, 'epochs': epochs,
         'dropout': dropout}, {'val_acc': max_val})


def build_model(dropout, hidden_dim):
    return MLP(784, 10, hidden_dim, dropout)


def build_optimizer(model, optimizer_name, learning_rate, weight_decay):
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=learning_rate, weight_decay=weight_decay)
        print("aici")
    elif optimizer_name == "adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(),
                                        lr=learning_rate, weight_decay=weight_decay, eps=1e-08)
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay)
    elif optimizer_name == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(),
                                        lr=learning_rate, eps=1e-08, weight_decay=weight_decay)
    elif optimizer_name == "sam":
        base_optimizer = torch.optim.SGD  # define an optimizer for the "sharpness-aware" update
        optimizer = SAM(model.parameters(), base_optimizer, lr=learning_rate, momentum=0.9)
    else:
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    return optimizer


def sweep_t(config=None, device=get_default_device()):
    main_transforms = [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.49139968, 0.48215827, 0.44653124], std=[0.24703233, 0.24348505, 0.26158768]),
        v2.Resize((28, 28), antialias=True),
        v2.Grayscale(),

    ]
    train_transforms = [
        v2.RandomAffine(degrees=10, translate=(0.0, 0.1), scale=(0.8, 1.0)),
        v2.RandomHorizontalFlip(0.25),
        v2.RandomVerticalFlip(0.25),
        torch.flatten,
    ]
    valid_transforms = [
        torch.flatten,
    ]
    # train_transforms = [
    # v2.RandomAffine(degrees=10, translate=(0.0, 0.1), scale=(0.8, 1.0)),
    # v2.RandomHorizontalFlip(0.25),
    # v2.RandomVerticalFlip(0.25),
    # v2.Normalize(mean=[0.49139968, 0.48215827 ,0.44653124], std=[0.24703233, 0.24348505, 0.26158768]),
    # v2.Resize((28, 28), antialias=True),
    # v2.Grayscale(),
    # torch.flatten,
    # ]
    # valid_transforms = [
    #     v2.Normalize(mean=[0.49139968, 0.48215827 ,0.44653124], std=[0.24703233, 0.24348505, 0.26158768]),
    #     v2.Resize((28, 28), antialias=True),
    #     v2.Grayscale(),
    #     torch.flatten,
    # ]

    data_path = '../data'
    train_dataset = CIFAR10(root=data_path, train=True, transform=v2.Compose(main_transforms), download=True)
    val_dataset = CIFAR10(root=data_path, train=False, transform=v2.Compose(main_transforms), download=True)
    train_dataset = CachedDataset(train_dataset, transforms=v2.Compose(train_transforms))
    val_dataset = CachedDataset(val_dataset, transforms=v2.Compose(valid_transforms))
    with wandb.init(config=config):
        config = wandb.config
        model = build_model(config.dropout, config.hidden_dim)
        model = model.to(device)
        optimizer = build_optimizer(model, config.optimizer, config.learning_rate, config.weight_decay)
        criterion = torch.nn.CrossEntropyLoss()
        epochs = config.epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
        batch_size = config.batch_size
        val_batch_size = 500
        num_workers = config.num_workers
        print(device)
        persistent_workers = (num_workers != 0)
        pin_memory = device.type == 'cuda'
        train_loader = DataLoader(train_dataset, shuffle=True, pin_memory=pin_memory, num_workers=num_workers,
                                  batch_size=batch_size, drop_last=True, persistent_workers=persistent_workers)
        val_loader = DataLoader(val_dataset, shuffle=False, pin_memory=True, num_workers=0, batch_size=val_batch_size,
                                drop_last=False)
        max_val = 0
        tbar = tqdm(tuple(range(epochs)))
        writer = None;
        for epoch in tbar:
            acc, acc_val, train_loss, val_loss = do_epoch(model, train_loader, val_loader, criterion, optimizer,
                                                          scheduler,
                                                          writer, epoch, device)
            if acc_val > max_val:
                max_val = acc_val
            tbar.set_postfix_str(f"Acc: {acc}, Acc_val: {acc_val}")
            wandb.log({"acc_val": acc_val, "train_loss": train_loss, "val_loss": val_loss, "epoch": epoch})
            # print("gata");


if __name__ == '__main__':
    freeze_support()
    wandb.login()
    sweep_config = {
        'method': 'bayes'
    }
    metric = {
        "name": "acc_val",
        "goal": "maximize"
    }
    parameters_dict = {
        "epochs": {
            "distribution": "int_uniform",
            "min": 50,
            "max": 500
        },
        "batch_size": {
            "distribution": "int_uniform",
            "min": 128,
            "max": 512
        },
        "hidden_dim": {
            "distribution": "int_uniform",
            "min": 128,
            "max": 2048
        },
        "optimizer": {
            "values": ["sgd", 'adam', 'adamw', 'adagrad', 'rmsprop']
        },
        "num_workers": {
            "distribution": "int_uniform",
            "min": 2,
            "max": 8
        },
        "weight_decay": {
            "distribution": "uniform",
            "min": 0.0025,
            "max": 0.01
        },
        "learning_rate": {
            "values": [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]
        },
        'dropout': {
            'values': [0.2, 0.3, 0.4, 0.5]
        },
    }

    sweep_config['parameters'] = parameters_dict

    sweep_config['metric'] = metric

    sweep_id = wandb.sweep(sweep_config, project="CIFAR10-low")
    wandb.agent(sweep_id, sweep_t, count=20)
    # start()