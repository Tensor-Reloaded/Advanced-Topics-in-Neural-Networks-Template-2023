import gc
from functools import wraps
from multiprocessing import freeze_support
from time import time

import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch
from torch import nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader, TensorDataset
from PIL import Image

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


class ImageTransform(nn.Module):
    def __init__(self):
        super(ImageTransform, self).__init__()
        self.fc = nn.Linear(3*32*32, 1*28*28)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        x = x.reshape(x.size(0), 1, 28, 28)
        return x

def accuracy(output, labels):
    fp_plus_fn = torch.logical_not(output == labels).sum().item()
    all_elements = len(output)
    return (all_elements - fp_plus_fn) / all_elements

def get_model_norm(model):
    norm = 0.0
    for param in model.parameters():
        norm += torch.norm(param)
    return norm

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class Pipeline:
    def __init__(self, model, train_loader, val_loader, optimizer, criterion, device, writer, is_sam, scheduler=None, early_stopper=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.writer = writer
        self.is_sam = is_sam
        self.early_stopper = early_stopper

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
            # print(f"Loss: {loss}")


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

            output = output.cpu().squeeze()
            labels = labels.cpu().squeeze()

            # print(f"Shapes: output {output.shape} / label {labels.shape}")
            all_outputs.append(output)
            all_labels.append(labels)
            all_losses.append(loss.item())

        all_outputs = torch.cat(all_outputs)
        all_labels = torch.cat(all_labels)

        return np.mean(all_losses)

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

                output = output.cpu().squeeze()
                labels = labels.cpu().squeeze()

                all_outputs.append(output)
                all_labels.append(labels)
                all_losses.append(loss.item())

        all_outputs = torch.cat(all_outputs)
        all_labels = torch.cat(all_labels)

        return np.mean(all_losses)

    def run(self, epochs):
        tbar = tqdm(range(epochs))
        for epoch in tbar:
            loss = self.train(epoch)
            loss_val = self.val(epoch)
            tbar.set_postfix_str(f"Loss: {loss}, Loss_val: {loss_val}")
            self.writer.add_scalar("Train/Loss", loss, epoch)
            self.writer.add_scalar("Val/Loss", loss_val, epoch)
            self.writer.add_scalar("Model/Norm", get_model_norm(self.model), epoch)

            if self.early_stopper.early_stop(loss_val):             
                break

def build_optimizer(model):
    is_sam = False

    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    # optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
    # base_optimizer = torch.optim.SGD
    # optimizer = SAM(model.parameters(), base_optimizer, lr=0.01, momentum=0.9)

    return optimizer, is_sam

def build_criterion(device):
    criterion = torch.nn.MSELoss()
    criterion = criterion.to(device)
    return criterion

def build_model(device):
    model = ImageTransform()
    model = model.to(device)
    return model

def build_dataloaders(train_dataset, val_dataset, pin_memory, num_workers, persistent_workers, batch_size, val_batch_size):
    train_loader = DataLoader(train_dataset, shuffle=True, pin_memory=pin_memory, num_workers=num_workers,
                            batch_size=batch_size, drop_last=True, persistent_workers=persistent_workers)
    val_loader = DataLoader(val_dataset, shuffle=False, pin_memory=True, num_workers=0, batch_size=val_batch_size,
                            drop_last=False)
    return train_loader, val_loader

def task_solving():
    device = get_default_device()

    epochs = 100
    batch_size = 256
    val_batch_size = 512
    num_workers = 5
    persistent_workers = (num_workers != 0)
    pin_memory = device.type == 'cuda'

    model = build_model(device)

    criterion = build_criterion(device)
    optimizer, is_sam = build_optimizer(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    early_stopper = EarlyStopper(patience=3, min_delta=10)

    writer = SummaryWriter()

    train_dataset = CustomDataset(train=True)
    val_dataset = CustomDataset(train=False)
    train_loader, val_loader = build_dataloaders(train_dataset, val_dataset, pin_memory, num_workers, persistent_workers, batch_size, val_batch_size)

    pipeline = Pipeline(model, train_loader, val_loader, optimizer, criterion, device, writer, is_sam=is_sam, scheduler=scheduler, early_stopper=early_stopper)
    pipeline.run(epochs) 

    torch.save(model.state_dict(), 'ImageTransform.pt')

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
def transform_dataset_with_model(dataset: CustomDataset, model: nn.Module, batch_size: int):
    model.eval()  # TODO: uncomment this
    dataloader = DataLoader(dataset, shuffle=False, pin_memory=True, num_workers=0, batch_size=batch_size,
                            drop_last=False)  # TODO: Complete the other parameters
    for images in dataloader:
        model(images[0])  # TODO: uncomment this
        # pass


def test_inference_time(model: nn.Module, device=torch.device('cpu')):
    test_dataset = CustomDataset(train=False)
    test_tensor_dataset = torch.stack(test_dataset.images)
    test_tensor_dataset = TensorDataset(test_tensor_dataset)

    batch_size = 512  # TODO: add the other parameters (device, ...)

    t1 = transform_dataset_with_transforms(test_tensor_dataset)
    t2 = transform_dataset_with_model(test_dataset, model, batch_size)
    print(f"Sequential transforming each image took: {t1} on CPU. \n"
          f"Using a model with batch_size: {batch_size} took {t2} on {device.type}. \n")


def download_examples():
    cifar10 = CIFAR10(root='./data', train=True, download=True, transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]))

    images = [img for img, _ in cifar10][:10]

    for i, img in enumerate(images):
        pil_img = v2.ToPILImage()(img)
        pil_img.save(f'image_{i}.jpg')

def task_generate_examples():
    model = ImageTransform()
    model.load_state_dict(torch.load('ImageTransform.pt'))

    transform = v2.Compose([
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True) 
    ])

    label_transform = v2.Compose([
        v2.Resize((28, 28), antialias=True),
        v2.Grayscale(),
        v2.functional.hflip,
        v2.functional.vflip,
    ])

    for idx in range(10):
        input_image = Image.open(f"./image_{idx}.jpg")

        # model
        input_tensor = transform(input_image).unsqueeze(0) 

        with torch.no_grad():
            output_tensor = model(input_tensor)

        output_tensor = output_tensor.squeeze(0)
        output_image = v2.ToPILImage()(output_tensor)
        output_image.save(f"./output_{idx}.jpg")

        # label
        label_image = label_transform(input_image)
        label_image.save(f"./label_{idx}.jpg")

def task_testing():
    model = ImageTransform()
    model.load_state_dict(torch.load('ImageTransform.pt'))

    device = get_default_device()

    test_inference_time(model=model, device=device)


if __name__ == '__main__':
    freeze_support()
    # task_solving() # train and save the model
    # download_examples() # save examples from cifar10 for compare
    # task_generate_examples() # generate examples with model
    task_testing() # test the model
