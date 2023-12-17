import torch
from torch import nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import gc
from functools import wraps
from multiprocessing import freeze_support
from time import time
import math
import random
import numpy as np
from PIL import Image
import wandb


global_transform = v2.Compose([
            v2.Resize((28, 28), antialias=True),
            v2.Grayscale(),
            v2.functional.hflip,
            v2.functional.vflip,
        ])


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
        # For multi-gpu workstations, PyTorch will use the first available GPU (cuda:0), unless specified otherwise
        # (cuda:1).
    if torch.backends.mps.is_available():
        return torch.device('mos')
    return torch.device('cpu')


def timed(fn: callable):
    @wraps(fn)
    def wrap(*args, **kwargs):
        gc.collect()
        start = time()
        fn(*args, **kwargs)
        end = time()
        return end - start

    return wrap

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc = nn.Linear(3*32*32, 1*28*28)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        x = x.reshape(x.size(0), 1, 28, 28)
        return x

def get_model_norm(model):
    norm = 0.0
    for param in model.parameters():
        norm += torch.norm(param)
    return norm

def get_cifar10_images(data_path: str, train: bool):
    initial_transforms = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    cifar_10_images = CIFAR10(root=data_path, train=train, transform=initial_transforms, download=True)
    return [image for image, label in cifar_10_images]


class CustomDataset(Dataset):
    def __init__(self, data_path: str = './data', train: bool = True, cache: bool = True):
        self.images = get_cifar10_images(data_path, train)
        self.cache = cache
        global global_transform
        self.transforms = global_transform
        if cache:
            self.labels = [self.transforms(x) for x in self.images]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        if self.cache:
            return self.images[i], self.labels[i]
        return self.images[i], self.transforms(self.images[i])

class MyEarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.min_loss_valid = math.inf
        self.count = 0
        self.patience = patience
        self.min_delta = min_delta

    def early_stopping(self, valid_loss):
        if self.min_loss_valid > valid_loss:
            self.min_loss_valid = valid_loss
            self.count = 0
        #daca nu se imbunatateste validation loss-ul de mai mult de un nr de ori opresc antrenarea
        elif (self.min_loss_valid + self.min_delta) < valid_loss:
            self.count = self.count + 1
            if self.count >= self.patience:
                return True
        return False
    

class MyTrainingPipeline:
    def __init__(self, model, train_loader, val_loader, optimizer, criterion, device, sum_writer, scheduler=None, early_stopper=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.sum_writer = sum_writer
        self.early_stopper = early_stopper

        self.sum_writer.add_scalar("Model/Learning Rate", self.optimizer.param_groups[0]["lr"])
        self.sum_writer.add_scalar("Model/Batch Size", self.train_loader.batch_size)
        self.sum_writer.add_text("Model/Optimizer", str(self.optimizer))

    def train_epoch(self, epoch):
        self.model.train()
        losses = []

        for i, batch_content in enumerate(self.train_loader):
            content = batch_content[0].to(self.device, non_blocking=True)
            labels = batch_content[1].to(self.device, non_blocking=True)
            output = self.model(content)
            loss = self.criterion(output, labels)
            self.sum_writer.add_scalar("Train/Batch Loss", loss.item(), epoch * len(self.train_loader) + i)
            self.sum_writer.add_scalar("Model/Learning Rate", self.optimizer.param_groups[0]["lr"], epoch * len(self.train_loader) + i)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            if self.scheduler is not None:
              self.scheduler.step()
            output = output.cpu().squeeze()
            labels = labels.cpu().squeeze()
            losses.append(loss.item())
    
        return np.mean(losses)

    def validation_epoch(self, epoch):
        self.model.eval()
        losses = []

        for i, batch_content in enumerate(self.val_loader):
            content = batch_content[0].to(self.device, non_blocking=True)
            labels = batch_content[1].to(self.device, non_blocking=True)
            with torch.no_grad():
                output = self.model(content)
                loss = self.criterion(output, labels)
                self.sum_writer.add_scalar("Val/Batch Loss", loss.item(), epoch * len(self.val_loader) + i)
                output = output.cpu().squeeze()
                labels = labels.cpu().squeeze()
                losses.append(loss.item())

        return np.mean(losses)

    def start_pipeline(self, epochs):
        for epoch in range(epochs):
            train_loss = self.train_epoch(epoch)
            valid_loss = self.validation_epoch(epoch)
            wandb.log({"Train Loss": train_loss, "Validation Loss": valid_loss})
            print(f"Epoch: {epoch}, Loss: {train_loss}, Loss_val: {valid_loss}")
            self.sum_writer.add_scalar("Train/Loss", train_loss, epoch)
            self.sum_writer.add_scalar("Val/Loss", valid_loss, epoch)
            self.sum_writer.add_scalar("Model/Norm", get_model_norm(self.model), epoch)

            if self.early_stopper.early_stopping(valid_loss):             
                break

def create_model():
    wandb.init(project="Homework7", settings=wandb.Settings(start_method="thread"))
    device = get_default_device()
    pin_memory = False
    if device.type == 'cuda':
        pin_memory = True

    #setez numarul de epoci, batchsize, etc..
    nr_epochs = 100
    batch_size = 256
    val_batch_size = 512
    num_workers = 5

    #construiesc modelul
    model = MyModel().to(device)

    #instantiez functia de loss
    criterion = torch.nn.MSELoss().to(device)

    #instantiez optimizer-ul
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    #Cosine Annealing Learning Rate Scheduler care scade lr-ul pe masura ce trec epocile
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=nr_epochs)

    #Early stopping
    early_stopper = MyEarlyStopper(patience=3, min_delta=10)
    sum_writer = SummaryWriter(f'C:\\Users\\mbrezuleanu\\runs\\logs')

    #construiesc dataload-urile     
    train_loader = DataLoader(CustomDataset(train=True), shuffle=True, pin_memory=pin_memory, num_workers=num_workers,
                            batch_size=batch_size, drop_last=True, persistent_workers=True)
    val_loader = DataLoader(CustomDataset(train=False), shuffle=False, pin_memory=True, num_workers=0, batch_size=val_batch_size,
                            drop_last=False)
    #instantiez clasa MyTrainingPipeline
    train_pipe = MyTrainingPipeline(model, train_loader, val_loader, optimizer, criterion, device, sum_writer, scheduler=cosine_scheduler, early_stopper=early_stopper)
    train_pipe.start_pipeline(nr_epochs) 
    
    #salvez modelul
    torch.save(model.state_dict(), 'MyModel.pt')
    wandb.finish()

@timed
def transform_dataset_with_transforms(dataset: TensorDataset):
    global global_transform
    transforms = global_transform
    for image in dataset.tensors[0]:
        transforms(image)


@timed
@torch.no_grad()
def transform_dataset_with_model(dataset: CustomDataset, model: nn.Module, batch_size: int):
    model.eval()  # TODO: uncomment this
    dataloader = DataLoader(dataset, shuffle=False, pin_memory=True, num_workers=0, batch_size=batch_size,
                            drop_last=False)  # TODO: Complete the other parameters
    for images in dataloader:
        model(images[0])  # TODO: uncomment this
        # pass


def test_inference_time(model: nn.Module, device=torch.device('cpu'), batch_sizes=None):
    test_dataset = CustomDataset(train=False)
    test_tensor_dataset = torch.stack(test_dataset.images)
    test_tensor_dataset = TensorDataset(test_tensor_dataset)

    if batch_sizes is None:
        batch_sizes = [512] 
    
    for batch_size in batch_sizes:
        t1 = transform_dataset_with_transforms(test_tensor_dataset)
        t2 = transform_dataset_with_model(test_dataset, model, batch_size)
        print(f"Sequential transforming each image took: {t1} on CPU. \n"
            f"Using a model with batch_size: {batch_size} took {t2} on {device.type}. \n")

def compare_times(batch_sizes=None):
    my_model = MyModel()
    my_model.load_state_dict(torch.load('MyModel.pt'))

    device = get_default_device()

    test_inference_time(model=my_model, device=device, batch_sizes=batch_sizes)

def extract_random_samples():
    cifar10 = CIFAR10(root='./data', train=True, download=True, transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]))
    random_images = random.sample(list(cifar10), 10)
    images = [img for img, _ in random_images]
    for i, img in enumerate(images):
        pil_img = v2.ToPILImage()(img)
        pil_img.save(f'in_img_{i}.jpg')

def save_images():
    #instantiez modelul meu si il incarc
    my_model = MyModel()
    my_model.load_state_dict(torch.load('MyModel.pt'))

    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True) ])
    global global_transform
    label_transform = global_transform

    for nr in range(10):
        sample = Image.open(f"./in_img_{nr}.jpg")
        in_tensor = transform(sample).unsqueeze(0) 

        with torch.no_grad():
            output_tensor = my_model(in_tensor).squeeze(0)

        out_image = v2.ToPILImage()(output_tensor)
        out_image.save(f"./out_img_{nr}.jpg")

        lab_image = label_transform(sample)
        lab_image.save(f"./lab_img_{nr}.jpg")

if __name__ == '__main__':
    freeze_support()
    create_model() 
    extract_random_samples() 
    save_images() 
    compare_times() 