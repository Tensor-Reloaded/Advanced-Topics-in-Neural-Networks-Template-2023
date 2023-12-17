import math
import os
from multiprocessing import freeze_support
from typing import List, Union
import gc
from time import time
from functools import wraps

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2
from tqdm import tqdm


def timed(fn):
    @wraps(fn)
    def wrap(*args, **kwargs):
        gc.collect()
        start = time()
        ret = fn(*args, **kwargs)
        end = time()
        return ret, end - start

    return wrap


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, cfg, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(cfg[0], cfg[1], kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cfg[1])
        self.conv2 = nn.Conv2d(cfg[1], cfg[2], kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(cfg[2])
        self.conv3 = nn.Conv2d(cfg[2], planes * 4, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class PreResNet(nn.Module):
    def __init__(self, depth=164):
        super(PreResNet, self).__init__()
        assert (depth - 2) % 9 == 0, 'depth should be 9n+2'

        n = (depth - 2) // 9
        block = Bottleneck

        cfg = [[16, 16, 16],
               [64, 16, 16] * (n - 1),
               [64, 32, 32],
               [128, 32, 32] * (n - 1),
               [128, 64, 64],
               [256, 64, 64] * (n - 1),
               [256]]
        cfg = [item for sub_list in cfg for item in sub_list]

        self.inplanes = 16
        self.fc = nn.Linear(cfg[-1], 10)
        in_channels = 3

        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 16, n, cfg=cfg[0:3 * n])
        self.layer2 = self._make_layer(block, 32, n, cfg=cfg[3 * n:6 * n], stride=2)
        self.layer3 = self._make_layer(block, 64, n, cfg=cfg[6 * n:9 * n], stride=2)
        self.bn = nn.BatchNorm2d(64 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)

        self.init_model()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()

    def init_model(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # nn.init.normal_(m.weight, 0, math.sqrt(2. / n))
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                # nn.init.constant_(m.weight, 0.5)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, cfg, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
            )

        layers = [block(self.inplanes, planes, cfg[0:3], stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cfg[3 * i: 3 * (i + 1)]))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8
        x = self.bn(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class CachedDataset(Dataset):
    def __init__(self, dataset: Dataset, runtime_transforms: List[Union[v2.Transform, nn.Module]], cache: bool):
        if cache:
            dataset = tuple([x for x in dataset])
        self.dataset = dataset
        if not len(runtime_transforms):
            runtime_transforms.append(nn.Identity())
        # If MonkeyType is not installed, do not install it.
        self.runtime_transforms = torch.jit.script(nn.Sequential(*runtime_transforms),
                                                   example_inputs=[(self.dataset[0][0],)])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        image, label = self.dataset[i]
        return self.runtime_transforms(image), label


def get_dataset(data_path: str, train: bool):
    initial_transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=(0.491, 0.482, 0.446),
            std=(0.247, 0.243, 0.261)
        ),
    ])
    cifar10 = CIFAR10(root=data_path, train=train, transform=initial_transforms, download=True)
    runtime_transforms = []
    if train:
        runtime_transforms = [
            v2.RandomCrop(size=32, padding=4),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.RandomErasing()
        ]
    return CachedDataset(cifar10, runtime_transforms, True)


@torch.jit.script
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
        output = model(data)
        loss = criterion(output, labels)

        loss.backward()

        # torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        output = output.softmax(dim=1).detach().cpu().squeeze()
        labels = labels.cpu().squeeze()
        all_outputs.append(output)
        all_labels.append(labels)

    all_outputs = torch.cat(all_outputs).argmax(dim=1)
    all_labels = torch.cat(all_labels)

    return round(accuracy(all_outputs, all_labels), 4)


def val(model, val_loader, device):
    model.eval()

    all_outputs = []
    all_labels = []

    for data, labels in val_loader:
        data = data.to(device, non_blocking=True)

        with torch.no_grad():
            output = model(data)

        output = output.softmax(dim=1).cpu().squeeze()
        labels = labels.squeeze()
        all_outputs.append(output)
        all_labels.append(labels)

    all_outputs = torch.cat(all_outputs).argmax(dim=1)
    all_labels = torch.cat(all_labels)

    return round(accuracy(all_outputs, all_labels), 4)


def do_epoch(model, train_loader, val_loader, criterion, optimizer, device):
    acc = train(model, train_loader, criterion, optimizer, device)
    acc_val = val(model, val_loader, device)
    # torch.cuda.empty_cache()
    return acc, acc_val


def main(device: torch.device = get_default_device(), data_path: str = './data', models_path: str = "./models"):
    os.makedirs(models_path, exist_ok=True)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    train_dataset = get_dataset(data_path, train=True)
    val_dataset = get_dataset(data_path, train=False)

    model = PreResNet(56)
    model = model.to(device)
    model = torch.jit.script(model, example_inputs=[(torch.rand((5, 3, 32, 32), device=device),)])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=10,
                                                           threshold=0.001, threshold_mode='rel')
    criterion = torch.nn.CrossEntropyLoss()

    batch_size = 50
    val_batch_size = 500
    num_workers = 0
    persistent_workers = (num_workers != 0) and False
    train_loader = DataLoader(train_dataset, shuffle=True, pin_memory=(device.type == 'cuda'), num_workers=num_workers,
                              batch_size=batch_size, drop_last=True, persistent_workers=persistent_workers)
    val_loader = DataLoader(val_dataset, shuffle=False, pin_memory=True, num_workers=0, batch_size=val_batch_size,
                            drop_last=False)

    tbar = tqdm(tuple(range(500)))
    best_val = 0.0
    for _ in tbar:
        acc, acc_val = do_epoch(model, train_loader, val_loader, criterion, optimizer, device)
        scheduler.step(acc)

        if acc_val > best_val:
            torch.save(model.state_dict(), os.path.join(models_path, "best.pth"))
            best_val = acc_val
        tbar.set_postfix_str(f"Acc: {acc}, Acc_val: {acc_val}, Best_val: {best_val}")


@timed
def infer(model, val_loader, device, tta, dtype):
    model.eval()
    all_outputs = []
    all_labels = []

    # Autocast is slow for cpu, so we disable it.
    # Autocast does not need to be used when using torch.float32
    # Also, if the device type is mps, autocast might not work (?) and disabling it might also not work (?)
    with torch.autocast(device_type=device.type, dtype=dtype, enabled=(device.type != 'cpu' or dtype != torch.float32)):
        for data, labels in val_loader:
            data = data.to(device, non_blocking=True)

            with torch.no_grad():
                output = model(data)
                if tta:
                    # Horizontal rotation:
                    output += model(v2.functional.hflip(data))
                    # Vertical rotation:
                    output += model(v2.functional.vflip(data))
                    # Horizontal rotation + Vertical rotation:
                    output += model(v2.functional.hflip(v2.functional.vflip(data)))

            output = output.softmax(dim=1).cpu().squeeze()
            labels = labels.squeeze()
            all_outputs.append(output)
            all_labels.append(labels)

    all_outputs = torch.cat(all_outputs).argmax(dim=1)
    all_labels = torch.cat(all_labels)

    return round(accuracy(all_outputs, all_labels), 4)


def predict(device: torch.device = get_default_device(), data_path: str = './data', models_path: str = "./models"):
    val_dataset = get_dataset(data_path, train=False)

    model = PreResNet(56)
    model = model.to(device)
    model.load_state_dict(torch.load(os.path.join(models_path, "best.pth"), map_location=device))

    val_batch_size = 500

    val_loader = DataLoader(val_dataset, shuffle=False, pin_memory=True, num_workers=0, batch_size=val_batch_size,
                            drop_last=False)

    # TODO: Check whether the dtypes and torch.compile are supported for your platform.
    if os.name == 'nt':
        # Windows is not supported, try Linux or WSL instead.
        torch.compile = lambda x: x

    for tta in (False, True):
        for dtype in (torch.bfloat16, torch.half, torch.float32):
            acc_val, seconds = infer(model, val_loader, device, tta=tta, dtype=dtype)
            print(f"Val acc: {acc_val}, tta: {tta}, dtype: {dtype}, took: {seconds}, raw model")
            acc_val, seconds = infer(torch.jit.script(model), val_loader, device, tta=tta, dtype=dtype)
            print(f"Val acc: {acc_val}, tta: {tta}, dtype: {dtype}, took: {seconds}, scripted model")
            acc_val, seconds = infer(torch.jit.trace(model, torch.rand((5, 3, 32, 32), device=device)), val_loader,
                                     device, tta=tta, dtype=dtype)
            print(f"Val acc: {acc_val}, tta: {tta}, dtype: {dtype}, took: {seconds}, traced model")
            acc_val, seconds = infer(torch.compile(model), val_loader, device, tta=tta, dtype=dtype)
            print(f"Val acc: {acc_val}, tta: {tta}, dtype: {dtype}, took: {seconds}, compiled model")
            print()


if __name__ == "__main__":
    freeze_support()
    main()
    predict()
