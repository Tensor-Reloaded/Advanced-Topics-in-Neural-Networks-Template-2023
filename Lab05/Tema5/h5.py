import os
from multiprocessing import freeze_support
import wandb
import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader
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
    def __init__(self, dataset, cache=True):
        if cache:
            dataset = tuple([x for x in dataset])
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return self.dataset[i]


class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size2, hidden_size3, hidden_size4, output_size):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size2)
        self.fc3 = torch.nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = torch.nn.Linear(hidden_size3, hidden_size4)
        self.fc5 = torch.nn.Linear(hidden_size4, output_size)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        return self.fc5(self.relu(self.fc4(self.relu(self.fc3(self.relu(self.fc2(self.relu(self.fc1(x)))))))))


def accuracy(output, labels):
    fp_plus_fn = torch.logical_not(output == labels).sum().item()
    all_elements = len(output)
    return (all_elements - fp_plus_fn) / all_elements


def train(model, train_loader, criterion, optimizer, epoch, writer, device):
    model.train()

    total_loss = 0.0

    all_outputs = []
    all_labels = []

    for batch_index, (data, labels) in enumerate(train_loader):
        data = data.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        output = model(data)

        def closure():
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, labels)
            loss.backward()
            closure.current_loss = loss.item()
            writer.add_scalar("Train/Batch_Loss", loss.item(), global_step=epoch * len(train_loader) + batch_index)
            return loss
        
        optimizer.step(closure)

        total_loss += closure.current_loss
        output = output.softmax(dim=1).detach().cpu().squeeze()
        labels = labels.cpu().squeeze()
        all_outputs.append(output)
        all_labels.append(labels)

    all_outputs = torch.cat(all_outputs).argmax(dim=1)
    all_labels = torch.cat(all_labels)

    return round(accuracy(all_outputs, all_labels), 4), total_loss/len(train_loader)


def val(model, val_loader, criterion, device):
    model.eval()

    total_loss = 0.0

    all_outputs = []
    all_labels = []

    for data, labels in val_loader:
        data = data.to(device, non_blocking=True)

        with torch.no_grad():
            output = model(data)

        loss = criterion(output, labels)
        total_loss += loss.item()

        output = output.softmax(dim=1).cpu().squeeze()
        labels = labels.squeeze()
        all_outputs.append(output)
        all_labels.append(labels)

    all_outputs = torch.cat(all_outputs).argmax(dim=1)
    all_labels = torch.cat(all_labels)

    return round(accuracy(all_outputs, all_labels), 4), total_loss/len(val_loader)


def do_epoch(model, train_loader, val_loader, criterion, optimizer, epoch, writer, device):
    acc_train, loss_train = train(model, train_loader, criterion, optimizer, epoch, writer, device)
    acc_val, loss_val = val(model, val_loader, criterion, device)
    # torch.cuda.empty_cache()
    return acc_train, loss_train, acc_val, loss_val


def get_model_norm(model):
    norm = 0.0
    for param in model.parameters():
        norm += torch.norm(param)
    return norm


def main_train(train_dataset, val_dataset, batch_size_val, learning_rate_val, optimizer_val, device=get_default_device()):
    
    #Create model
    model = MLP(784, 3500, 2048, 1024, 512, 10)
    model = model.to(device)
    
    #Define optimizer
    if optimizer_val == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate_val)
    elif optimizer_val == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_val)
    elif optimizer_val == "sgd_with_sam":
        base_optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate_val)
        optimizer = SAM(model.parameters(), base_optimizer, rho=0.05)  
    elif optimizer_val == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate_val)
    elif optimizer_val == "adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate_val)

    criterion = torch.nn.CrossEntropyLoss()

    #Define Data Loaders
    epochs = 75
    validation_batch_size = 500
    num_workers = 2
    persistent_workers = (num_workers != 0)
    pin_memory = device.type == 'cuda'
    train_loader = DataLoader(train_dataset, shuffle=True, pin_memory=pin_memory, num_workers=num_workers,
                              batch_size=batch_size_val, drop_last=True, persistent_workers=persistent_workers)
    val_loader = DataLoader(val_dataset, shuffle=False, pin_memory=True, num_workers=0, batch_size=validation_batch_size,
                            drop_last=False)
    writer = SummaryWriter(f'C:\\Users\\mbrezuleanu\\runs\\logs\\batch_size {batch_size_val} lr {learning_rate_val} optimizer {optimizer_val}')

    for epoch in range(epochs):
        acc_train, loss_train, acc_val, loss_val = do_epoch(model, train_loader, val_loader, criterion, optimizer, epoch, writer, device)
        writer.add_scalar("Train/Accuracy", acc_train, epoch)
        writer.add_scalar("Val/Accuracy", acc_val, epoch)
        writer.add_scalar("Train/Loss", loss_train, epoch)
        writer.add_scalar("Val/Loss", loss_val, epoch)
        writer.add_scalar("Model/Norm", get_model_norm(model), epoch)
        writer.add_scalar("Batch Size", batch_size_val, epoch)
        writer.add_scalar('Learning Rate', learning_rate_val, epoch)
        writer.add_text('Optimizer', optimizer.__class__.__name__, epoch)
        wandb.log({"Train Loss": loss_train, "Validation Loss": loss_val, "Accuracy Train": acc_train, "Accuracy Validation": acc_val})


def set_config():
    sweep_config = {
    'method': 'grid'
    }
    metric = {
    'name': 'accuracy',
    'goal': 'maximize'   
    }
    sweep_config['metric'] = metric
    parameters_dict = {
    'optimizer': {
        'values': ['sgd','adam', 'adagrad', 'rmsprop','sgd_with_sam']
        },
    'batch_size': {
        'values': [64]
        },
    'learning_rate': {
          'values': [0.001, 0.01, 0.1]
        },
    }

    sweep_config['parameters'] = parameters_dict
    sweep_id = wandb.sweep(sweep_config, project="homework5")
    return sweep_id

def apply_config(config=None):
    transforms = [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((28, 28), antialias=True),
        v2.Grayscale(),
        torch.flatten,
    ]

    data_path = r'D:\\Facultate\\Master1\\ACNN\\Teme\\Homework5'
    train_dataset = CIFAR10(root=data_path, train=True, transform=v2.Compose(transforms), download=True)
    val_dataset = CIFAR10(root=data_path, train=False, transform=v2.Compose(transforms), download=True)
    train_dataset = CachedDataset(train_dataset)
    val_dataset = CachedDataset(val_dataset)


    # Initialize a new wandb run
    with wandb.init(config=config):
        config = wandb.config
        main_train(train_dataset, val_dataset, config.batch_size, config.learning_rate, config.optimizer)

if __name__ == '__main__':
    freeze_support()
    sweep_id = set_config()
    wandb.agent(sweep_id, apply_config)

