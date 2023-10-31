import torch
from torch.utils.data import DataLoader

from Assignment5.dataset import Dataset
from Assignment5.loader import TrainLoader, TestLoader, ValidateLoader
from Assignment5.model import Model
from Assignment5.train import TrainTune
from Assignment5.utils import split_dataset, split_dataset_csv


class Runner:  # corresponds to scenario when dataset needs to be loaded and processed from local device, as we work;
    # for including other various scenarios a super class might be built
    def __init__(self, epochs: int, device: torch.device('cpu'),
                 dataset_path, dataset_builder,
                 similarity_func=torch.nn.CosineSimilarity(), treshold=0.95):
        self.dataset_builder = dataset_builder
        self.epochs = epochs
        self.device = device
        self.dataset_path = dataset_path
        self.similarity = similarity_func
        self.treshold = treshold

    def run_model(self, model: Model, dataset_csv: bool = False, split_path='',
                  transforms=None, transforms_test=None, pin_memory=False):
        # set pin_memory as true if GPU is not used for transformations
        urban_dataset = Dataset(self.dataset_path, self.dataset_builder)

        if not dataset_csv:
            split_dataset(urban_dataset, '.\\Homework_Dataset_Split')
            train_dataset = Dataset(f'{split_path}\\train', self.dataset_builder,
                                    transformations=transforms, transformations_test=transforms_test)
            test_dataset = Dataset(f'{split_path}\\test', self.dataset_builder,
                                   transformations=transforms, transformations_test=transforms_test,
                                   training=False)
            validation_dataset = Dataset(f'{split_path}\\validate', self.dataset_builder,
                                         transformations=transforms, transformations_test=transforms_test,
                                         training=False)

            train_loader = DataLoader(train_dataset, batch_size=32,
                                      shuffle=True, pin_memory=pin_memory)
            test_loader = DataLoader(test_dataset, batch_size=32,
                                     shuffle=False, pin_memory=pin_memory)
            validation_loader = DataLoader(validation_dataset, batch_size=32,
                                           shuffle=False, pin_memory=pin_memory)
        else:
            split_dataset_csv(urban_dataset)
            train_loader = TrainLoader('train.csv', transformations=transforms,
                                       transformations_test=transforms_test,
                                       batch_size=32, shuffle=True, pin_memory=pin_memory)
            test_loader = TestLoader('test.csv', transformations=transforms, batch_size=32,
                                     transformations_test=transforms_test,
                                     shuffle=False, pin_memory=pin_memory)
            validation_loader = ValidateLoader('validation.csv', transformations=transforms,
                                               transformations_test=transforms_test,
                                               batch_size=32, shuffle=False, pin_memory=pin_memory)
        train_tune = TrainTune(model, train_loader, validation_loader, device=self.device,
                               similarity=self.similarity, treshold=self.treshold)
        train_tune.run(self.epochs)
