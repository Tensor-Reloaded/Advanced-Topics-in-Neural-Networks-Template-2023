from multiprocessing import freeze_support
import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2, Normalize, RandomHorizontalFlip, RandomCrop
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as f
from tqdm import tqdm
import time


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
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


class SimpleCNN(torch.nn.Module):
    def __init__(self, img_width, img_height):
        super(SimpleCNN, self).__init__()

        input_width = img_width
        input_height = img_height

        channels = 3
        nr_conv_filters = 3
        conv_filter_size = 5
        pool_size = 4
        output_size = 10

        self.convLayer = torch.nn.Conv2d(channels, nr_conv_filters, kernel_size=conv_filter_size)
        self.poolLayer = torch.nn.MaxPool2d(pool_size)
        fc_input_size = (input_height - pool_size * (conv_filter_size // pool_size)) * (
                input_width - pool_size * (conv_filter_size // pool_size)) * nr_conv_filters // (pool_size * pool_size)
        self.fcLayer = torch.nn.Linear(fc_input_size, output_size)

    def forward(self, input_image):
        output = self.convLayer(input_image)
        output = self.poolLayer(output)
        output = f.relu(output)
        output = output.view([1, -1])
        output = self.fcLayer(output)

        return output


def accuracy(output, labels):
    fp_plus_fn = torch.logical_not(output == labels).sum().item()
    all_elements = len(output)
    return (all_elements - fp_plus_fn) / all_elements


def train(model, train_loader, criterion, optimizer, device):
    model.train()

    all_outputs = []
    all_labels = []

    for data, labels in train_loader:
        data = data.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        for image, label in zip(data, labels):
            optimizer.zero_grad()
            output = model(image.unsqueeze(0))
            loss = criterion(output, label.unsqueeze(0))
            loss.backward()
            optimizer.step()

            output = output.softmax(dim=1).detach().cpu().squeeze()
            predicted_label = torch.argmax(output)
            all_outputs.append(predicted_label.item())
        all_labels.append(labels)

    all_labels = torch.cat(all_labels)
    all_outputs = torch.as_tensor(all_outputs)
    all_outputs = all_outputs.to(device, non_blocking=True)

    return round(accuracy(all_outputs, all_labels), 4)


def val(model, val_loader, device):
    model.eval()

    all_outputs = []
    all_labels = []

    for data, labels in val_loader:
        data = data.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        for image, label in zip(data, labels):
            output = model(image.unsqueeze(0))

            output = output.softmax(dim=1).detach().cpu().squeeze()
            predicted_label = torch.argmax(output)
            all_outputs.append(predicted_label.item())
        all_labels.append(labels)

    all_labels = torch.cat(all_labels)
    all_outputs = torch.as_tensor(all_outputs)
    all_outputs = all_outputs.to(device, non_blocking=True)

    return round(accuracy(all_outputs, all_labels), 4)


def do_epoch(model, train_loader, val_loader, criterion, optimizer, device):
    acc = train(model, train_loader, criterion, optimizer, device)
    acc_val = val(model, val_loader, device)
    return acc, acc_val


def main(device=get_default_device()):
    print('Device used:', device)
    transforms = [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        RandomHorizontalFlip(),
        RandomCrop(size=32),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]

    data_path = '../data'
    train_dataset = CIFAR10(root=data_path, train=True, transform=v2.Compose(transforms), download=True)
    val_dataset = CIFAR10(root=data_path, train=False, transform=v2.Compose(transforms), download=True)
    train_dataset = CachedDataset(train_dataset)
    val_dataset = CachedDataset(val_dataset)

    batch_size = 256
    val_batch_size = 500
    num_workers = 2
    persistent_workers = (num_workers != 0)
    pin_memory = device.type == 'cuda'
    train_loader = DataLoader(train_dataset, shuffle=True, pin_memory=pin_memory, num_workers=num_workers,
                              batch_size=batch_size, drop_last=True, persistent_workers=persistent_workers)
    val_loader = DataLoader(val_dataset, shuffle=False, pin_memory=True, num_workers=0, batch_size=val_batch_size,
                            drop_last=False)

    model = SimpleCNN(32, 32)
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    epochs = 20
    learn_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    tbar = tqdm(tuple(range(epochs)))
    for epoch in tbar:
        print('Epoch:', epoch + 1)
        acc, acc_val = do_epoch(model, train_loader, val_loader, criterion, optimizer, device)
        tbar.set_postfix_str(f"Acc: {acc}, Acc_val: {acc_val}")


if __name__ == '__main__':
    start = time.time()

    freeze_support()
    main()

    end = time.time()
    print('Needed time for execution:', end - start)
