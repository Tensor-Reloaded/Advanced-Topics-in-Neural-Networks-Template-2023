import random
import gc
import time
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from functools import wraps
from data import CustomDataset


def ground_truth_transform(image):
    transformed_image = F.resize(image, size=(28, 28), interpolation=InterpolationMode.BILINEAR, antialias=True)
    transformed_image = F.rgb_to_grayscale(transformed_image, num_output_channels=1)
    transformed_image = F.hflip(transformed_image)
    transformed_image = F.vflip(transformed_image)
    return transformed_image


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.patience_counter = 0

    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                return True
        return False


def timed(fn: callable):
    @wraps(fn)
    def wrap(*args, **kwargs):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.time()
        result = fn(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.time()
        return result, end - start

    return wrap


def generate_and_save_images(model, dataset, num_images=5, device=torch.device('cpu')):
    model.eval()
    selected_indices = random.sample(range(len(dataset)), num_images)
    for idx in selected_indices:
        image, _ = dataset[idx]  # Get image and ignore label

        # Move image to the same device as the model
        image = image.to(device)

        # Process image with the model
        with torch.no_grad():
            processed_img = model(image.unsqueeze(0)).squeeze(0)  # Add and remove batch dimension

        # Generate ground truth image
        ground_truth_img = ground_truth_transform(image)

        # Save images
        save_image(processed_img.cpu(), f'processed_image_{idx}.png')
        save_image(ground_truth_img.cpu(), f'ground_truth_image_{idx}.png')


@timed
@torch.no_grad()
def infer_and_compare(model, dataset, device):
    model.eval()
    model.to(device)
    differences = []
    for i in range(len(dataset)):
        original, ground_truth = dataset[i]
        original = original.unsqueeze(0).to(device)
        ground_truth = ground_truth.unsqueeze(0).to(device)
        transformed = model(original)
        diff = nnf.mse_loss(transformed, ground_truth)
        differences.append(diff.item())

    avg_difference = sum(differences) / len(differences)

    return avg_difference


@timed
def transform_dataset_with_transforms(dataloader: DataLoader):
    transformed_images = []
    for images, _ in dataloader:
        for image in images:
            transformed_images.append(ground_truth_transform(image))
    return torch.stack(transformed_images)


@timed
@torch.no_grad()
def transform_dataset_with_model(dataloader: DataLoader, model: nn.Module, device: torch.device):
    model.eval()
    model.to(device)
    transformed_images = []
    for images, _ in dataloader:
        images = images.to(device)
        transformed = model(images)
        transformed_images.append(transformed.cpu())
    return torch.cat(transformed_images, dim=0)


def test_inference_time(model: nn.Module, device: torch.device, batch_size: int):
    test_dataset = CustomDataset(data_path='./data', train=False, transform=ground_truth_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Transform with traditional methods
    _, sequential_time = transform_dataset_with_transforms(test_loader)

    # Transform with model
    _, model_batch_time = transform_dataset_with_model(test_loader, model, device)

    return sequential_time, model_batch_time
