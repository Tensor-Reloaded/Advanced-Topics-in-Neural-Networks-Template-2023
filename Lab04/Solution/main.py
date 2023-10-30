import torch.nn

from trainingPipeline import *
from datasets import *
import torchvision.transforms.v2 as transforms


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
        # For multi-gpu workstations, PyTorch will use the first available GPU (cuda:0), unless specified otherwise
        # (cuda:1).
    if torch.backends.mps.is_available():
        return torch.device('mos')
    return torch.device('cpu')


def main():
    device = get_default_device()

    image_size = (32, 32)
    transformers = []
    transformers.append(transforms.Compose([transforms.Grayscale(), transforms.Compose(
        [transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)])]))
    transformers.append(transforms.Compose([transforms.Resize(image_size, antialias=True), transforms.Compose(
        [transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)])]))

    dataset = MosaicDataset(device, dataset_folder="Homework Dataset", use_random_rotation=True,
                            transformers=transformers)

    # Parameters for datasets
    dataset_proportions = (0.7, 0.15, 0.15)
    batch_size = 32
    shuffle = True

    training_pipeline = TrainingPipeline(device, dataset, dataset_proportions, batch_size, shuffle)

    # Parameters for the model and training
    input_dimension = image_size[0] * image_size[1] + 1
    output_dimension = image_size[0] * image_size[1]
    no_units_per_layer = [input_dimension, output_dimension]
    output_activation = torch.nn.Identity()

    no_epochs = 1

    training_pipeline.run(no_epochs, no_units_per_layer, output_activation)


if __name__ == '__main__':
    main()
