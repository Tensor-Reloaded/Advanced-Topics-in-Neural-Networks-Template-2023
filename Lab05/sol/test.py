import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import v2

import torch
import torch.nn as nn
import torch.optim as optim

import logging
import os
import wandb

from torch.utils.tensorboard import SummaryWriter


class CIFAR10LOAD:
    def __init__(self, seed, dataset_path, train_ratio=0.8, batch_size=32, device=None):
        self.loaded_dataset, self.test_loader, self.train_loader, self.train_set, self.test_set = [None] * 5
        self.train_ratio = train_ratio
        self.batch_size = batch_size
        self.device_type = device
        self.seed = seed
        self.dataset_path = dataset_path
        self.transforms = [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((28, 28), antialias=True),
            v2.Grayscale(),
            torch.flatten,
        ]
        self.dataset_loaded = False

    def load_datasets(self):
        if self.dataset_loaded:
            return self.loaded_dataset
        transform = torchvision.transforms.Compose(self.transforms)
        self.loaded_dataset = torchvision.datasets.CIFAR10(root=self.dataset_path, train=True, download=True, transform=transform)
        self.dataset_loaded = True
        return self.train_set, self.test_set

    def get_data_loaders(self):
        if self.train_loader is not None and self.test_loader is not None:
            return self.train_loader, self.test_loader

        self.load_datasets()

        train_size = int(self.train_ratio * len(self.loaded_dataset))
        test_size = len(self.loaded_dataset) - train_size

        train_dataset, test_dataset = random_split(self.loaded_dataset, [train_size, test_size])
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size)

        return self.train_loader, self.test_loader





class C10Class:
    def __init__(self, data_loader_builder, config, log_file=None, device=None):
        self.log_file = log_file
        self.config = config

        self.logger = self.initialize_logger()

        # Load CIFAR-10 dataset
        self.data_loader = data_loader_builder
        self.train_d, self.test_d = self.data_loader.get_data_loaders()

        self.device_type = device
        self.device = Utils.initialize_device(self.device_type)

        self.model = self.initialize_model()

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self.initialize_optimizer()

    def initialize_model(self):
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128, self.config['model_norm']),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64, self.config['model_norm']),
            nn.Linear(64, 10)
        )

        return model.to(self.device)

    def initialize_optimizer(self):
        optimizer_name = self.config['optimizer']
        lr = self.config['learning_rate']
        optimizer_dict = {
            'SGD': optim.SGD(self.model.parameters(), lr=lr),
            'Adam': optim.Adam(self.model.parameters(), lr=lr),
            'RMSprop': optim.RMSprop(self.model.parameters(), lr=lr),
            'Adagrad': optim.Adagrad(self.model.parameters(), lr=lr)
        
        }

        return optimizer_dict.get(optimizer_name, None)

    def initialize_logger(self):
        logs = logging.getLogger('C10Class')
        logs.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        if self.log_file:
            if not os.path.exists(os.path.dirname(self.log_file)):
                os.makedirs(os.path.dirname(self.log_file))
            my_file = logging.FileHandler(self.log_file)
            my_file.setFormatter(formatter)
            logs.addHandler(my_file)

        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logs.addHandler(console)

        return logs

    def train(self):
        writer_train = SummaryWriter('tensorboard_logs_out/train')

        total_train_loss = 0.0

        for epoch in range(self.config['num_epochs']):
            total_epoch_loss = 0.0
            total_batches = len(self.train_d)

            for i, data in enumerate(self.train_d, 0):
                inputs, labels = data

                self.optimizer.zero_grad()

                outputs = self.model(inputs.to(self.device))

                loss = self.criterion(outputs, labels.to(self.device))
                last_batch_loss = loss.item()

                wandb.log({'epoch': epoch, 'batch_nr': i, 'avg_batch_loss': last_batch_loss})
                loss.backward()
                self.optimizer.step()

                total_epoch_loss += loss.item()

            average_batch_loss = total_epoch_loss / total_batches

            writer_train.add_scalar('Loss/Train', average_batch_loss, epoch)
            wandb.log({'epoch': epoch, 'loss': average_batch_loss})
            self.logger.info('Epoch %d - Epoch Loss: %f', epoch, average_batch_loss)

            total_train_loss += average_batch_loss

            wandb.log({'epoch': epoch, 'avg_epoch_loss': total_train_loss / (epoch + 1)})
            writer_train.add_scalar('Epoch Loss/Train', total_train_loss / (epoch + 1))

        writer_train.add_scalar('Loss/Train', total_train_loss / self.config['num_epochs'])

        writer_train.close()

    def evaluate(self):
        self.model.eval()
        test_loss = 0.0
        good = 0
        total = 0

        with torch.no_grad():
            for data in self.test_d:
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                test_loss += loss.item()

                _, predicted = outputs.max(1)
                total += labels.size(0)
                good += predicted.eq(labels).sum().item()

        test_accuracy = good / total * 100
        test_loss /= len(self.test_d)
        self.model.train()

        eval = SummaryWriter('tensorboard_logs_out/eval')
        eval.add_scalar('Loss/Test', test_loss)
        eval.add_scalar('Accuracy/Test', test_accuracy)
        eval.add_hparams({k: v for k, v in self.config.items() if k != 'device'},{'hparam/loss': test_loss,'hparam/accuracy': test_accuracy},)
        eval.close()

        wandb.log({'test_loss': test_loss, 'test_accuracy': test_accuracy})
        return test_loss, test_accuracy
    

import torch


class Utils:
    @staticmethod
    def initialize_device(device=None):
        if device is not None:
            return torch.device(device)
        else:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        

DATASET_PATH = './dataset'

#RMSprop, Adagrad, Adam, SGD
config = {
        'seed': 100,
        'dataset_path': './dataset',
        'device': 'cpu',
        'train_percentage': 0.8,
        'learning_rate': 0.01,
        'optimizer': 'Adagrad',
        'num_epochs': 10,
        'model_norm': 0.9,
        'batch_size': 32,
}

wandb.init(project="lab5", config=config)

data_loaders_builder = CIFAR10LOAD(seed=config['seed'], dataset_path=DATASET_PATH, train_ratio=config['train_percentage'], device=config['device'], batch_size=config['batch_size'])
train_d, test_d = data_loaders_builder.get_data_loaders()

pipeline = C10Class(
        data_loader_builder=data_loaders_builder,
        device=config['device'],
        config=config,
        log_file='./logg2.txt'
)

pipeline.train()
pipeline.evaluate()
wandb.finish()