import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from Assignment4.model import Model
from Assignment4.plotter import MetricsMemory


class TrainTune:
    def __init__(self, cmodel: Model,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 similarity=torch.nn.CosineSimilarity(),
                 device: torch.device = torch.device('cpu')):
        self.model = cmodel
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizers = []
        self.device = device
        self.similarity = similarity
        if cmodel.optimizers and len(cmodel.optimizers) > 0:
            for index, optimizer in enumerate(cmodel.optimizers):
                if cmodel.optimizer_args and len(cmodel.optimizer_args) > index \
                        and cmodel.optimizer_args[index] != {}:
                    self.optimizers.append(optimizer(self.model.parameters(),
                                                     **(cmodel.optimizer_args[index])))
                else:
                    self.optimizers.append(optimizer(nn.ParameterList(self.model.parameters()),
                                                     **cmodel.default_optim_args))

    def train(self):
        self.model.train()
        total_loss = 0
        correct = 0
        pbar = tqdm(total=len(self.train_loader),
                    desc="Training", dynamic_ncols=True)
        for features, labels in self.train_loader:
            features = features.to(self.device)
            for index, optimizer in enumerate(self.optimizers):
                self.optimizers[index].zero_grad()
            outputs = self.model(features)
            labels = labels.to(self.device)
            correct += (self.similarity(outputs, labels) > 0.95).sum()
            loss = self.model.loss(outputs, labels)
            loss.backward()
            for index, optimizer in enumerate(self.optimizers):
                self.optimizers[index].step()
            total_loss += loss.item()
            pbar.set_postfix({'Loss': loss.item()})
            pbar.update()
        pbar.close()
        return total_loss, correct

    def val(self):
        self.model.eval()
        correct = 0
        total_loss = 0
        with torch.no_grad():
            for features, labels in self.val_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(features)
                correct += (self.similarity(outputs, labels) > 0.95).sum()
                total_loss += self.model.loss(outputs, labels).item()
        return total_loss, correct

    def run(self, n: int):
        metrics_memory = MetricsMemory(n)
        for epoch in range(n):
            total_loss, correct_train = self.train()
            valid_loss, correct_test = self.val()
            train_metrics = (correct_train / (len(self.train_loader) * self.train_loader.batch_size),
                             total_loss / (len(self.train_loader) * self.train_loader.batch_size))
            val_metrics = (correct_test / (len(self.val_loader) * self.val_loader.batch_size),
                           valid_loss / (len(self.val_loader) * self.val_loader.batch_size))
            metrics_memory.update_metrics(epoch, val_metrics, train_metrics)
        metrics_memory.draw_plot()
