import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam, RMSprop, Adagrad
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import wandb

# Function to get default device
def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

def get_model_norm(model):
    norm = 0.0
    for param in model.parameters():
        norm += torch.norm(param)
    return norm

# Function to train the model for one epoch
def train(model, train_loader, criterion, optimizer, device, wandb, epoch):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    for data, labels in tqdm(train_loader, desc=f"Epoch {epoch}"):
        data = data.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = output.max(1)
        correct_predictions += predicted.eq(labels).sum().item()
        total_samples += labels.size(0)

    epoch_loss = total_loss / len(train_loader)
    epoch_accuracy = correct_predictions / total_samples

    wandb.log({"Loss/Train": epoch_loss, "Accuracy/Train": epoch_accuracy, "epoch": epoch})

# Function to validate the model for one epoch
def validate(model, val_loader, criterion, device, wandb, epoch):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for data, labels in tqdm(val_loader, desc=f"Validation {epoch}"):
            data = data.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            output = model(data)
            loss = criterion(output, labels)

            total_loss += loss.item()
            _, predicted = output.max(1)
            correct_predictions += predicted.eq(labels).sum().item()
            total_samples += labels.size(0)

    epoch_loss = total_loss / len(val_loader)
    epoch_accuracy = correct_predictions / total_samples

    wandb.log({"Loss/Validation": epoch_loss, "Accuracy/Validation": epoch_accuracy, "epoch": epoch})

# Main function for training and evaluation
def main(device=get_default_device()):
    input_size = 28 * 28
    hidden_size = 100
    output_size = 10
    model = MLP(input_size, hidden_size, output_size).to(device)

    wandb.init(project='atnn-lab05', name='experiment_1', config={'lr': 0.01, 'batch_size': 256, 'val_batch_size': 512, 'optimizer': 'RMSprop'})

    wandb.watch(model, log="all")

    tb_writer = SummaryWriter()

    transforms = [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((28, 28), antialias=True),
        v2.Grayscale(),
        torch.flatten,
    ]

    train_dataset = CIFAR10(root='./data', train=True, transform=v2.Compose(transforms), download=True)
    val_dataset = CIFAR10(root='./data', train=False, transform=v2.Compose(transforms), download=True)

    train_loader = DataLoader(train_dataset, batch_size=wandb.config.batch_size*2, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=wandb.config.val_batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # Choose optimizer and learning rate from W&B config
    optimizer_config = wandb.config.optimizer
    if optimizer_config == 'SGD':
        optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4, nesterov=True)
    elif optimizer_config == 'Adam':
        optimizer = Adam(model.parameters(), lr=0.001)
    elif optimizer_config == 'RMSprop':
        optimizer = RMSprop(model.parameters(), lr=0.001, weight_decay=1e-4)
    elif optimizer_config == 'Adagrad':
        optimizer = Adagrad(model.parameters(), lr=0.01, weight_decay=1e-4)
    else:
        raise ValueError(f"Invalid optimizer: {optimizer_config}")

    criterion = CrossEntropyLoss()

    epochs = 20
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0

        # for batch_idx, (data, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
        #     data = data.to(device, non_blocking=True)
        #     labels = labels.to(device, non_blocking=True)

        #     optimizer.zero_grad()
        #     output = model(data)
        #     loss = criterion(output, labels)
        #     loss.backward()
        #     optimizer.step()

        #     total_loss += loss.item()
        #     _, predicted = output.max(1)
        #     correct_predictions += predicted.eq(labels).sum().item()
        #     total_samples += labels.size(0)

        #     tb_writer.add_scalar('Batch/Training Loss', loss.item(), epoch * len(train_loader) + batch_idx)

        # epoch_loss = total_loss / len(train_loader)
        # epoch_accuracy = correct_predictions / total_samples
        # tb_writer.add_scalar('Epoch/Training Loss', epoch_loss, epoch)
        # tb_writer.add_scalar('Epoch/Training Accuracy', epoch_accuracy, epoch)

        # validate(model, val_loader, criterion, device, wandb, epoch)

        # tb_writer.add_scalar('Epoch/Validation Loss', wandb.config.lr, epoch)
        # tb_writer.add_scalar('Epoch/Validation Accuracy', epoch_accuracy, epoch)
        # tb_writer.add_scalar('Others/Model Norm', torch.norm(torch.cat([param.view(-1) for param in model.parameters()])), epoch)
        # tb_writer.add_scalar('Others/Learning Rate', wandb.config.lr, epoch)
        # tb_writer.add_scalar('Others/Batch Size', wandb.config.batch_size, epoch)

    tb_writer.close()
    wandb.log({"Learning Rate": wandb.config.lr, "Optimizer": optimizer_config})

    
if __name__ == '__main__':
    main()
