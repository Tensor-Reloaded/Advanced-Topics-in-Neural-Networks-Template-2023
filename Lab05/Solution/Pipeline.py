from multiprocessing import freeze_support
import wandb
import torch
import pprint
import numpy as np
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

def wandb_login():
    wandb.login()

def build_sweep_config():
    sweep_config = {
        'method': 'grid',
        'metric': {
            'name': 'loss',
            'goal': 'minimize'
        },
        'parameters': {
            'optimizer': {
                'values': ['adam', 'adamw', 'sgd', 'rmsprop', 'adagrad', 'sam']
            },
            'epochs': {
                'value': 10
            },
            'learning_rate': {
                'values': [0.01, 0.001, 0.0001]
            },
            'batch_size': {
                'values': [256]
            },
            'val_batch_size': {
                'values': [512]
            }
        }
    }
    pprint.pprint(sweep_config)
    return sweep_config

def sweep_id(sweep_config):
    sweep_id = wandb.sweep(sweep_config, project="Homework5_Solution")
    return sweep_id

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
        if self.transforms is not None:
          return (self.transforms(self.dataset[i][0]),self.dataset[i][1])
        else:
          return self.dataset[i]

class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size[0]),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_size[0], hidden_size[1]),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_size[1], output_size)
        )
        for layer in self.layers:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        x = self.layers(x)
        return x
    
class EvolvedMLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(EvolvedMLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size[0])
        self.relu = torch.nn.ReLU(inplace=True)
        self.fc2 = torch.nn.Linear(hidden_size[0], hidden_size[1])
        self.fc3 = torch.nn.Linear(hidden_size[1], hidden_size[2])
        self.dropout = torch.nn.Dropout(0.3)
        self.fc4 = torch.nn.Linear(hidden_size[2], output_size)
        
        torch.nn.init.kaiming_uniform_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.fc3.weight, mode='fan_in', nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.fc4.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        x1 = self.relu(self.fc1(x))
        x2 = self.relu(self.fc2(x1))
        x3 = self.relu(self.dropout(self.fc3(x2 + x))) # skip connections
        x4 = self.fc4(x3)
        return x4

def accuracy(output, labels):
    fp_plus_fn = torch.logical_not(output == labels).sum().item()
    all_elements = len(output)
    return (all_elements - fp_plus_fn) / all_elements

def get_model_norm(model):
    norm = 0.0
    for param in model.parameters():
        norm += torch.norm(param)
    return norm

class Pipeline:
    def __init__(self, model, train_loader, val_loader, optimizer, criterion, device, writer, is_wandb, is_sam, scheduler=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.writer = writer
        self.is_wandb = is_wandb
        self.is_sam = is_sam

        self.writer.add_scalar("Model/Learning Rate", self.optimizer.param_groups[0]["lr"])
        self.writer.add_scalar("Model/Batch Size", self.train_loader.batch_size)
        self.writer.add_text("Model/Optimizer", str(self.optimizer))

    def train(self, epoch):
        self.model.train()

        all_outputs = []
        all_labels = []
        all_losses = []

        for batch_idx, data_batch in enumerate(self.train_loader):
            data = data_batch[0].to(self.device, non_blocking=True)
            labels = data_batch[1].to(self.device, non_blocking=True)

            output = self.model(data)
            loss = self.criterion(output, labels)

            self.writer.add_scalar("Train/Batch Loss", loss.item(), epoch * len(self.train_loader) + batch_idx)
            self.writer.add_scalar("Model/Learning Rate", self.optimizer.param_groups[0]["lr"], epoch * len(self.train_loader) + batch_idx)

            loss.backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

            if self.is_sam:
              self.optimizer.first_step(zero_grad=True)
              self.criterion(self.model(data), labels).backward()
              self.optimizer.second_step(zero_grad=True)
            else:
              self.optimizer.step()
              self.optimizer.zero_grad(set_to_none=True)

            if self.scheduler is not None:
              self.scheduler.step()

            output = output.softmax(dim=1).detach().cpu().squeeze()
            labels = labels.cpu().squeeze()

            all_outputs.append(output)
            all_labels.append(labels)
            all_losses.append(loss.item())

        all_outputs = torch.cat(all_outputs).argmax(dim=1)
        all_labels = torch.cat(all_labels)

        if self.is_wandb:
          wandb.log({"Train Loss": np.mean(all_losses), "Epoch": epoch})
        self.writer.add_scalar("Train/Epoch Loss", np.mean(all_losses), epoch)

        return round(accuracy(all_outputs, all_labels), 4)

    def val(self, epoch):
        self.model.eval()

        all_outputs = []
        all_labels = []
        all_losses = []

        for batch_idx, data_batch in enumerate(self.val_loader):
            data = data_batch[0].to(self.device, non_blocking=True)
            labels = data_batch[1].to(self.device, non_blocking=True)

            with torch.no_grad():
                output = self.model(data)
                loss = self.criterion(output, labels)

                self.writer.add_scalar("Val/Batch Loss", loss.item(), epoch * len(self.val_loader) + batch_idx)

                output = output.softmax(dim=1).cpu().squeeze()
                labels = labels.cpu().squeeze()

                all_outputs.append(output)
                all_labels.append(labels)
                all_losses.append(loss.item())

        all_outputs = torch.cat(all_outputs).argmax(dim=1)
        all_labels = torch.cat(all_labels)

        if self.is_wandb:
          wandb.log({"Val Loss": np.mean(all_losses), "Epoch": epoch})
        self.writer.add_scalar("Val/Epoch Loss", np.mean(all_losses), epoch)

        return round(accuracy(all_outputs, all_labels), 4)

    def run(self, epochs):
        tbar = tqdm(range(epochs))
        for epoch in tbar:
            acc = self.train(epoch)
            acc_val = self.val(epoch)
            tbar.set_postfix_str(f"Acc: {acc}, Acc_val: {acc_val}")
            if self.is_wandb:
              wandb.log({"Train Acc": acc, "Epoch": epoch})
              wandb.log({"Val Acc": acc_val, "Epoch": epoch})
            self.writer.add_scalar("Train/Accuracy", acc, epoch)
            self.writer.add_scalar("Val/Accuracy", acc_val, epoch)
            self.writer.add_scalar("Model/Norm", get_model_norm(self.model), epoch)

def build_transforms():
    transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        # v2.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        v2.Resize((28, 28), antialias=True),
        v2.Grayscale(),
        v2.Normalize((0.4733,), (0.2515,)),
        torch.flatten
    ])

    train_transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        # v2.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        v2.Resize((28, 28), antialias=True)
    ])

    rand_transforms = v2.Compose([
        v2.RandAugment(),
        v2.Grayscale(),
        v2.Normalize((0.4733,), (0.2515,)),
        torch.flatten,
    ])

    return transforms, train_transforms, rand_transforms

def build_dataset():
    transforms, train_transforms, rand_transforms = build_transforms()

    data_path = '../data'
    train_dataset = CIFAR10(root=data_path, train=True, transform=train_transforms, download=True)
    val_dataset = CIFAR10(root=data_path, train=False, transform=transforms, download=True)
    train_dataset = CachedDataset(train_dataset, rand_transforms)
    val_dataset = CachedDataset(val_dataset)

    return train_dataset, val_dataset

def build_optimizer(model):
    is_sam = True

    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    # optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
    base_optimizer = torch.optim.SGD
    optimizer = SAM(model.parameters(), base_optimizer, lr=0.01, momentum=0.9)

    return optimizer, is_sam

def build_optimizer_wandb(parameters, optimizer, learning_rate):
    if optimizer == 'sgd':
        optimizer = torch.optim.SGD(parameters, lr=learning_rate, momentum=0.9)
    elif optimizer == 'adam':
        optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    elif optimizer == 'adamw':
        optimizer = torch.optim.AdamW(parameters, lr=learning_rate)
    elif optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(parameters, lr=learning_rate)
    elif optimizer == 'adagrad':
        optimizer = torch.optim.Adagrad(parameters, lr=learning_rate)
    elif optimizer == 'sam':
        base_optimizer = torch.optim.SGD
        optimizer = SAM(parameters, base_optimizer, lr=learning_rate, momentum=0.9)
    return optimizer

def build_criterion(device):
    criterion = torch.nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    return criterion

def build_model(device, input_size, hidden_size, output_size):
    model = MLP(input_size, hidden_size, output_size)
    model = model.to(device)
    return model

def build_evolved_model(device, input_size, hidden_size, output_size):
    model = EvolvedMLP(input_size, hidden_size, output_size)
    model = model.to(device)
    return model

def build_dataloaders(train_dataset, val_dataset, pin_memory, num_workers, persistent_workers, batch_size, val_batch_size):
    train_loader = DataLoader(train_dataset, shuffle=True, pin_memory=pin_memory, num_workers=num_workers,
                            batch_size=batch_size, drop_last=True, persistent_workers=persistent_workers)
    val_loader = DataLoader(val_dataset, shuffle=False, pin_memory=True, num_workers=0, batch_size=val_batch_size,
                            drop_last=False)
    return train_loader, val_loader

def tune(config=None):
    device = get_default_device()

    num_workers = 5
    persistent_workers = (num_workers != 0)
    pin_memory = device.type == 'cuda'
    input_size = 784
    hidden_size = [400, 784, 100] #[512, 256]
    output_size = 10

    # model = build_model(device, input_size, hidden_size, output_size)
    model = build_evolved_model(device, input_size, hidden_size, output_size)

    criterion = build_criterion(device)

    train_dataset, val_dataset = build_dataset()
    writer = SummaryWriter()

    with wandb.init(config=config):
        config = wandb.config
        train_loader = DataLoader(train_dataset, shuffle=True, pin_memory=pin_memory, num_workers=num_workers,
                                    batch_size=config.batch_size, drop_last=True, persistent_workers=persistent_workers)
        val_loader = DataLoader(val_dataset, shuffle=False, pin_memory=True, num_workers=0, batch_size=config.val_batch_size,
                                drop_last=False)
        optimizer = build_optimizer_wandb(model.parameters(), config.optimizer, config.learning_rate)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
        pipeline = Pipeline(model, train_loader, val_loader, optimizer, criterion, device, writer, is_wandb=True, is_sam=(config.optimizer=='sam'))
        pipeline.run(config.epochs)

def train_wandb(sweep_id):
    wandb.agent(sweep_id, function=tune)

def task_searching():
    wandb_login()
    sweep_config = build_sweep_config()
    id = sweep_id(sweep_config)

    train_wandb(id)

def task_solving():
    device = get_default_device()

    epochs = 200
    batch_size = 256
    val_batch_size = 512
    num_workers = 5
    persistent_workers = (num_workers != 0)
    pin_memory = device.type == 'cuda'
    input_size = 784
    hidden_size = [400, 784, 100] #[512, 256]
    output_size = 10

    # model = build_model(device, input_size, hidden_size, output_size)
    model = build_evolved_model(device, input_size, hidden_size, output_size)

    criterion = build_criterion(device)
    optimizer, is_sam = build_optimizer(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    writer = SummaryWriter()

    train_dataset, val_dataset = build_dataset()
    train_loader, val_loader = build_dataloaders(train_dataset, val_dataset, pin_memory, num_workers, persistent_workers, batch_size, val_batch_size)

    pipeline = Pipeline(model, train_loader, val_loader, optimizer, criterion, device, writer, is_wandb=False, is_sam=is_sam, scheduler=scheduler)
    pipeline.run(epochs)    

if __name__ == '__main__':
    freeze_support()
    task_solving() # try the best configurations
    # task_searching() # rank configurations
    
    





















