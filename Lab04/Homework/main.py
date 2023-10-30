import torch
import torch.nn as nn
from torch import optim
from chart import draw_chart
from pipeline import Pipeline
from dataset import SpacenetDataset
from network import PredictionModel
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomAdjustSharpness, GaussianBlur, Normalize

class ToFloat:
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return image.to(torch.float32)

def run_on_device(device):
    if device == 'cpu':
        return torch.device('cpu')
    elif device == 'gpu':
        if torch.cuda.is_available():
            return torch.device('cuda:0')
        else:
            print("GPU not available. Using CPU instead.")
            return torch.device('cpu')
    else:
        print("Invalid device parameter. Using CPU by default.")
        return torch.device('cpu')

def main():
    run_on_device('cpu')

    dataset_transformations = Compose([
        GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
        RandomHorizontalFlip(p=0.5),
        RandomAdjustSharpness(0.5, p=0.5),
        ToFloat(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = SpacenetDataset('../Homework Dataset', dataset_transformations, 'cpu')
    print(f"Dataset size: {len(dataset):,}")
    print(f"Image shape: {dataset.get_image_shape()}")

    train_dataset, validation_dataset, test_dataset = dataset.get_data_loaders()
    model = PredictionModel()

    model_loss_function = nn.MSELoss()
    model_optimizer = optim.Adam(model.parameters(), lr=0.001)

    pipeline = Pipeline(model, model_optimizer, model_loss_function, train_dataset, validation_dataset, test_dataset)
    train_losses, validation_losses = pipeline.run(10)

    draw_chart("Training loss", range(1, len(train_losses)+1), train_losses)
    draw_chart("Validation losses", range(1, len(validation_losses)+1), validation_losses)


if __name__ == "__main__":
    main()