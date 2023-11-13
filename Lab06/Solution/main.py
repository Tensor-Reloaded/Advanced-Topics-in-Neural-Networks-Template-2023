from multiprocessing import freeze_support

import torch
import wandb
# from google.colab import userdata
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2
from tqdm import tqdm


class WbaLogger:
    def __init__(self, optimizer_name, **options):
        # secret = userdata.get('CLIENT_SECRET')
        custom_str = ""
        for option in options:
            custom_str = f"{custom_str}_{option}={options[option]}"
        wandb.init(
            # set the wandb project where this run will be logged
            project="advanced_topics_lab6",
            name=f"{optimizer_name}_{custom_str}",
            # track hyperparameters and run metadata
            config={
                "learning_rate": options.get("lr", 0.01),
                "architecture": "MLP",
                "dataset": "CIFAR-100",
                "epochs": options.get("epochs", 50),
                "batch_size": options.get("batch_size", 512)
            }
        )

    def info(self, data):
        wandb.log(data)


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
        self.input_size = input_size

        layers = [input_size] + [hidden_size, hidden_size // 2, hidden_size // 4, hidden_size // 8] + [output_size]

        self.fcts = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])
        self.relu = torch.nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        # fwds = []
        # for size in range(0, 784, 28):
        #     fwds.append(x[:, size: size + 28])
        #
        #
        # res = []
        # for fwd in fwds:
        #     result = self.middle_fct(fwd)
        #     res.append(result)
        #
        # x = torch.concat([x, *res], dim=1)
        for fct_index, fct in enumerate(self.fcts):
            x = fct(x)
            if fct_index != len(self.fcts) - 1:
                x = self.relu(x)

        return x


def accuracy(output, labels):
    fp_plus_fn = torch.logical_not(output == labels).sum().item()
    all_elements = len(output)
    return (all_elements - fp_plus_fn) / all_elements


def train(model, train_loader, criterion, optimizer, device, writer, logger):
    model.train()

    all_outputs = []
    all_labels = []
    final_loss = 0
    for data, labels in train_loader:
        data = data.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        output = model(data)
        loss = criterion(output, labels)
        writer.add_scalar("batch_loss_train", loss.item())
        logger.info({"batch_loss_train": loss.item()})
        final_loss += loss.item()

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
    writer.add_scalar("epoch_loss_train", final_loss)
    logger.info({"epoch_loss_train": final_loss})

    return round(accuracy(all_outputs, all_labels), 4)


def val(model, val_loader, criterion, device, writer, logger):
    model.eval()

    all_outputs = []
    all_labels = []
    final_loss = 0
    for data, labels in val_loader:
        data = data.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.no_grad():
            output = model(data)

        loss = criterion(output, labels)
        logger.info({"batch_loss_val": loss.item()})
        writer.add_scalar("batch_loss_val",  loss.item())

        final_loss += loss.item()

        output = output.softmax(dim=1).cpu().squeeze()
        labels = labels.cpu().squeeze()

        all_outputs.append(output)
        all_labels.append(labels)

    all_outputs = torch.cat(all_outputs).argmax(dim=1)
    all_labels = torch.cat(all_labels)
    logger.info({"epoch_loss_val": final_loss})
    writer.add_scalar("epoch_loss_val", final_loss)

    return round(accuracy(all_outputs, all_labels), 4)


def do_epoch(model, train_loader, val_loader, criterion, optimizer, device, writer, logger):
    acc = train(model, train_loader, criterion, optimizer, device, writer, logger)
    acc_val = val(model, val_loader, criterion, device, writer, logger)
    # torch.cuda.empty_cache()
    return acc, acc_val


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
        torch.flatten,
    ]

    data_path = '../../data'
    train_dataset = CIFAR10(root=data_path, train=True, transform=v2.Compose(transforms), download=True)
    val_dataset = CIFAR10(root=data_path, train=False, transform=v2.Compose(transforms), download=True)
    train_dataset = CachedDataset(train_dataset)
    val_dataset = CachedDataset(val_dataset)

    model = MLP(784, 256, 10)
    model = model.to(device)

    lr = 0.01
    momentum = 0.9
    weight_decay = 1e-4
    batch_size = 512
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    print(str(optimizer).split(" (")[0])
    logger = WbaLogger(str(optimizer).split(" (")[0], lr=lr, momentum=momentum, weight_decay=weight_decay, batch_size=batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    epochs = 50

    batch_size = batch_size
    val_batch_size = 500
    num_workers = 2
    persistent_workers = (num_workers != 0)
    pin_memory = device.type == 'cuda'
    train_loader = DataLoader(train_dataset, shuffle=True, pin_memory=pin_memory, num_workers=num_workers,
                              batch_size=batch_size, drop_last=True, persistent_workers=persistent_workers)
    val_loader = DataLoader(val_dataset, shuffle=False, pin_memory=True, num_workers=0, batch_size=val_batch_size,
                            drop_last=False)

    writer = SummaryWriter()
    tbar = tqdm(tuple(range(epochs)))
    for epoch in tbar:
        acc, acc_val = do_epoch(model, train_loader, val_loader, criterion, optimizer, device, writer, logger)
        # scheduler_warmup.step()
        tbar.set_postfix_str(f"Acc: {acc}, Acc_val: {acc_val}")
        writer.add_scalar("Train/Accuracy", acc, epoch)
        writer.add_scalar("Val/Accuracy", acc_val, epoch)
        model_norm = get_model_norm(model)
        writer.add_scalar("Model/Norm", model_norm, epoch)
        writer.add_scalar("Learning rate", lr)
        writer.add_scalar("Batch size", batch_size)
        # writer.add_scalar("Optimizer", "SGD")

        logger.info({
            "acc": acc,
            "acc_val": acc_val,
            "model_norm": model_norm
        })
    wandb.finish()


if __name__ == '__main__':
    freeze_support()
    main()



