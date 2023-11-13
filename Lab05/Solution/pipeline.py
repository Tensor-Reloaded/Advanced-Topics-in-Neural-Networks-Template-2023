import torch
import torch.nn as nn
import torch.optim as optim

import logging
import os
import wandb

from torch.utils.tensorboard import SummaryWriter
from utils import Utils


class CIFAR10Pipeline:
    def __init__(self, data_loader_builder, config, log_file=None, device=None):
        self.log_file = log_file
        self.config = config

        self.logger = self.initialize_logger()

        # Load CIFAR-10 dataset
        self.data_loader = data_loader_builder
        self.trainset, self.testset = self.data_loader.get_data_loaders()

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
        if self.config['optimizer'] == 'SGD':
            return optim.SGD(self.model.parameters(), lr=self.config['learning_rate'])
        elif self.config['optimizer'] == 'Adam':
            return optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        elif self.config['optimizer'] == 'RMSprop':
            return optim.RMSprop(self.model.parameters(), lr=self.config['learning_rate'])
        elif self.config['optimizer'] == 'Adagrad':
            return optim.Adagrad(self.model.parameters(), lr=self.config['learning_rate'])
        # elif self.config['optimizer'] == 'SGD_DAM':
        #
        #     sgd_optimizer = optim.SGD(self.model.parameters(), lr=self.config['learning_rate'])
        #     return optim.SAM(sgd_optimizer, rho=0.05, adaptive=False)

    def initialize_logger(self):
        logger = logging.getLogger('CIFAR10Pipeline')
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        if self.log_file:
            # Log to a file
            if not os.path.exists(os.path.dirname(self.log_file)):
                os.makedirs(os.path.dirname(self.log_file))
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        # Log to the console
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        return logger

    def train(self):
        writer_train = SummaryWriter('tensorboard_logs_out/train')

        total_train_loss = 0.0

        for epoch in range(self.config['num_epochs']):
            total_epoch_loss = 0.0  # Initialize the epoch loss accumulator
            total_batches = len(self.trainset)

            for i, data in enumerate(self.trainset, 0):
                inputs, labels = data

                self.optimizer.zero_grad()

                outputs = self.model(inputs.to(self.device))

                loss = self.criterion(outputs, labels.to(self.device))
                last_batch_loss = loss.item()

                wandb.log({'epoch': epoch, 'batch_nr': i, 'avg_batch_loss': last_batch_loss})

                # spammy - loss / batch
                # self.logger.info('Epoch %d - Batch %d - Avg batch loss: %f', epoch, i, loss)
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

        writer_train.add_scalar('Final Loss/Train', total_train_loss / self.config['num_epochs'])

        writer_train.close()

    def evaluate(self):
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data in self.testset:
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                test_loss += loss.item()

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        test_loss /= len(self.testset)
        test_accuracy = 100 * correct / total

        self.model.train()

        writer_eval = SummaryWriter('tensorboard_logs_out/eval')

        writer_eval.add_scalar('Loss/Test', test_loss)
        writer_eval.add_scalar('Accuracy/Test', test_accuracy)

        writer_eval.add_hparams(
            {k: v for k, v in self.config.items() if k != 'device'},
            {'hparam/loss': test_loss,
             'hparam/accuracy': test_accuracy},
                                )

        writer_eval.close()

        wandb.log({'test_loss': test_loss, 'test_accuracy': test_accuracy})

        return test_loss, test_accuracy
