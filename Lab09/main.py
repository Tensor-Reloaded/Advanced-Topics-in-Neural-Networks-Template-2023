import gc
from functools import wraps
from multiprocessing import freeze_support
from time import time

import torch
import torchvision
from torch import nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader, TensorDataset


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
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


class SimpleModel(torch.nn.Module):
    def __init__(self, img_width, img_height):
        super(SimpleModel, self).__init__()

        input_width = img_width
        input_height = img_height

        in_channels_1 = 3
        in_channels_2 = 32
        nr_conv_filters_1 = 32
        nr_conv_filters_2 = 64
        conv_filter_size = 3
        pool_size = 2
        output_size1 = 1024
        output_size2 = 784

        self.convLayer1 = torch.nn.Conv2d(in_channels_1, nr_conv_filters_1, kernel_size=conv_filter_size)
        self.convLayer2 = torch.nn.Conv2d(in_channels_2, nr_conv_filters_2, kernel_size=conv_filter_size)
        self.poolLayer = torch.nn.MaxPool2d(kernel_size=pool_size, stride=2)
        fc_input_size = nr_conv_filters_2 * (((input_height - pool_size * (conv_filter_size // pool_size)) // pool_size -
                        pool_size * (conv_filter_size // pool_size)) // pool_size) * (((input_width - pool_size *
                        (conv_filter_size // pool_size)) // pool_size - pool_size * (conv_filter_size // pool_size)) // pool_size)

        self.fcLayer1 = torch.nn.Linear(fc_input_size, output_size1)
        self.fcLayer2 = torch.nn.Linear(output_size1, output_size2)

        self.activation_function = torch.nn.LeakyReLU(inplace=True)

        self.norm_1 = torch.nn.InstanceNorm2d(32)
        self.norm_2 = torch.nn.InstanceNorm2d(64)

        nn.init.kaiming_normal(self.fcLayer1.weight)
        nn.init.kaiming_normal(self.fcLayer2.weight)

    def forward(self, input_image):
        output = self.convLayer1(input_image)
        output = self.norm_1(output)
        output = self.activation_function(output)
        output = self.poolLayer(output)

        output = self.convLayer2(output)
        output = self.norm_2(output)
        output = self.activation_function(output)
        output = self.poolLayer(output)

        output = output.view([1, -1])

        output = self.fcLayer1(output)
        output = self.activation_function(output)

        output = self.fcLayer2(output)
        output = self.activation_function(output)

        output = output.view(1, 28, 28)

        return output


def get_ground_truth(dataset: TensorDataset):
    transforms = v2.Compose([
        v2.Resize((28, 28), antialias=True),
        v2.Grayscale(),
        v2.functional.hflip,
        v2.functional.vflip,
    ])
    ground_truth = []
    for image in dataset.tensors[0]:
        ground_truth.append(transforms(image))
    return ground_truth


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
def transform_dataset_with_model(dataset: TensorDataset, labels, model: nn.Module, batch_size: int, epochs, criterion,
                                 optimizer):
    model.train()
    previous_loss = 0
    for e in range(epochs):
        dataloader = DataLoader(dataset, batch_size=batch_size)

        batch_index = 0
        total_loss = 0
        for images in dataloader:
            item_index = 0
            for item in images[0]:
                output = model.forward(item)
                loss = criterion(output, labels[100 * batch_index + item_index])
                total_loss += loss
                optimizer.step()

                item_index += 1
            batch_index += 1

        if e == 0:
            previous_loss = total_loss
        elif e > 7:
            if abs(total_loss - previous_loss) < 10:
                print('Stop training')
                break
            else:
                previous_loss = total_loss


def test_inference_time(model: nn.Module, device=torch.device('cpu')):
    test_dataset = CustomDataset(train=False, cache=False)

    test_dataset = torch.stack(test_dataset.images)
    test_dataset = TensorDataset(test_dataset)

    batch_size = 100
    epochs = 3
    criterion = torch.nn.MSELoss()
    learn_rate = 0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    labels = get_ground_truth(test_dataset)

    t1 = transform_dataset_with_transforms(test_dataset)
    t2 = transform_dataset_with_model(test_dataset, labels, model, batch_size, epochs, criterion, optimizer)
    print(f"Sequential transforming each image took: {t1} on CPU. \n"
          f"Using a model with batch_size: {batch_size} took {t2} on {device.type}. \n")


def main(device=get_default_device()):
    print('Device used:', device)
    my_model = SimpleModel(32, 32)
    test_inference_time(my_model)


if __name__ == '__main__':
    freeze_support()
    main()
