import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR100, CIFAR10
from torchvision.transforms import v2

from Assignment6.dataset import Dataset
from Assignment6.loader import TrainLoader, TestLoader, ValidateLoader
from Assignment6.model import Model
from Assignment6.train import TrainTune
from Assignment6.utils import split_dataset, split_dataset_csv


class Runner:  # corresponds to scenario when dataset needs to be loaded and processed from local device, as we work;
    # for including other various scenarios a super class might be built
    def __init__(self, epochs: int, device: torch.device('cpu'),
                 dataset_path, dataset_builder, writer: SummaryWriter,
                 similarity_func=torch.nn.CosineSimilarity(),
                 treshold=0.95):
        self.dataset_builder = dataset_builder
        self.epochs = epochs
        self.device = device
        self.dataset_path = dataset_path
        self.similarity = similarity_func
        self.treshold = treshold
        self.writer = writer
        self.sweep_config = {
            'method': 'random'
        }
        self.configure_sweep()

    def run_model(self, model: Model, dataset_csv: bool = False, split_path='',
                  load_from_pytorch=False, transforms=None, transforms_test=None,
                  pin_memory=False, batch_size=64, val_batch_size=128, num_workers=2, persistent_workers=True,
                  transforms_not_cached=None, config=None, num_classes=10, model_forward=None):
        # set pin_memory as true if GPU is not used for transformations

        if not dataset_csv:
            if load_from_pytorch:
                if len(transforms) > 0:
                    train_dataset = CIFAR100(root='../data', train=True,
                                             transform=v2.Compose(transforms), download=True)
                else:
                    train_dataset = CIFAR100(root='../data', train=True, download=True)
                if len(transforms_test) > 0:
                    val_dataset = CIFAR100(root='../data', train=False,
                                           transform=v2.Compose(transforms_test), download=True)
                else:
                    val_dataset = CIFAR100(root='../data', train=False, download=True)
                train_dataset = Dataset('../data',
                                        lambda path: (tuple([x for x in train_dataset]), len(train_dataset)),
                                        transformations=transforms_not_cached, transformations_test=[])
                val_dataset = Dataset('../data',
                                      lambda path: (tuple([x for x in val_dataset]), len(val_dataset)),
                                      transformations=[], transformations_test=[], training=False)
                train_loader = DataLoader(train_dataset, shuffle=True, pin_memory=pin_memory,
                                          num_workers=num_workers, persistent_workers=persistent_workers,
                                          batch_size=batch_size, drop_last=True)
                validation_loader = DataLoader(val_dataset, shuffle=False, pin_memory=True, num_workers=0,
                                               batch_size=val_batch_size, drop_last=False)
            else:
                urban_dataset = Dataset(self.dataset_path, self.dataset_builder)
                split_dataset(urban_dataset, '.\\Homework_Dataset_Split')
                train_dataset = Dataset(f'{split_path}\\train', self.dataset_builder,
                                        transformations=transforms, transformations_test=transforms_test)
                test_dataset = Dataset(f'{split_path}\\test', self.dataset_builder,
                                       transformations=transforms, transformations_test=transforms_test,
                                       training=False)
                validation_dataset = Dataset(f'{split_path}\\validate', self.dataset_builder,
                                             transformations=transforms, transformations_test=transforms_test,
                                             training=False)

                train_loader = DataLoader(train_dataset, batch_size=64,
                                          shuffle=True, pin_memory=pin_memory)
                test_loader = DataLoader(test_dataset, batch_size=32,
                                         shuffle=False, pin_memory=pin_memory)
                validation_loader = DataLoader(validation_dataset, batch_size=128,
                                               shuffle=False, pin_memory=pin_memory)
        else:
            urban_dataset = Dataset(self.dataset_path, self.dataset_builder)
            split_dataset_csv(urban_dataset)
            train_loader = TrainLoader('train.csv', transformations=transforms,
                                       transformations_test=transforms_test,
                                       batch_size=64, shuffle=True, pin_memory=pin_memory)
            test_loader = TestLoader('test.csv', transformations=transforms, batch_size=32,
                                     transformations_test=transforms_test,
                                     shuffle=False, pin_memory=pin_memory)
            validation_loader = ValidateLoader('validation.csv', transformations=transforms,
                                               transformations_test=transforms_test,
                                               batch_size=128, shuffle=False, pin_memory=pin_memory)
        train_tune = TrainTune(model, train_loader, validation_loader, self.writer,
                               device=self.device, similarity=self.similarity, treshold=self.treshold,
                               config=config, no_class=num_classes, model_forward=model_forward)
        train_tune.run(self.epochs)

    def configure_sweep(self):
        metric = {
            'name': 'accuracy',
            'goal': 'maximize'
        }
        self.sweep_config['metric'] = metric
        parameters_dict = {
            'batch_size': {
                'values': [32, 64, 128]
            },
            'optimizer': {
                'values': ['adam', 'sgd', 'rmsprop', 'adagrad', 'sam_sgd']
            },
            'fc_layer_size_1': {
                'values': [4000, 1024, 1000, 512, 256, 128]
            },
            'fc_layer_size_2': {
                'values': [4000, 1024, 1000, 512, 256, 128]
            },
            'fc_layer_size_3': {
                'values': [4000, 1024, 1000, 512, 256, 128]
            },
            'dropout_1': {
                'values': [('a', 0.0), ('a', 0.3), ('b', 0.4), ('b', 0.5),
                           ('b', 0.0), ('b', 0.3), ('a', 0.4), ('a', 0.5)]
            },
            'dropout_2': {
                'values': [('a', 0.0), ('a', 0.3), ('b', 0.4), ('b', 0.5),
                           ('b', 0.0), ('b', 0.3), ('a', 0.4), ('a', 0.5)]
            },
            'dropout_3': {
                'values': [('a', 0.0), ('a', 0.3), ('b', 0.4), ('b', 0.5),
                           ('b', 0.0), ('b', 0.3), ('a', 0.4), ('a', 0.5)]
            },
            'dropout_4': {
                'values': [('a', 0.0), ('a', 0.3), ('b', 0.4), ('b', 0.5),
                           ('b', 0.0), ('b', 0.3), ('a', 0.4), ('a', 0.5)]
            },
            'batch_norm': {
                'values': [True, False]
            },
            'weight_decay': {
                'values': [0.0, 0.1, 0.001, 0.005, 0.0005, 0.00001]
            },
            'momentum': {
                'values': [0.0, 0.9]
            },
            'Nesterov': {
                'values': [True, False]
            },
            'lr': {
                'values': [0.1, 0.001, 0.005, 0.0005, 0.00001]
            },
            'gradient_clipping': {
                'values': [True, False]
            },
            'lr_scheduler': {
                'values': [True, False]
            },
        }
        self.sweep_config['parameters'] = parameters_dict
        parameters_dict.update({
            'epochs': {
                'value': self.epochs}
        })
