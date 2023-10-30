import torch
from torch import optim, nn

from Dataset import Dataset
from Grapher import Grapher
from ImagePredictionModel import ImagePredictionModel
from Pipeline import Pipeline
from Transform import Transform


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


if __name__ == '__main__':
    run_on_device('cpu')
    transform = Transform()
    dataset = Dataset('cpu', r'./Homework Dataset', transform.transforms)
    # dataset = Dataset('cpu', r'./Homework Dataset')
    train_dataloader, val_dataloader, test_dataloader = dataset.get_data_loaders()
    model = ImagePredictionModel()
    pipeline = Pipeline(model, optim.Adam(model.parameters(), lr=0.001), nn.MSELoss(), train_dataloader, val_dataloader,
                        test_dataloader, 'cpu')
    train_losses, val_losses = pipeline.run(5)
    grapher = Grapher()
    grapher.draw('Train Loss', range(0, len(train_losses)), train_losses)
    grapher.draw('Validation Loss', range(0, len(val_losses)), val_losses)
