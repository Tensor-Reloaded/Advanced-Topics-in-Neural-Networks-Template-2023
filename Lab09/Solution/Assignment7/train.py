import torch
import wandb
from torch import nn
from torch.utils.benchmark import timer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from Assignment7.model import Model
from Assignment7.plotter import MetricsMemory


class TrainTune:
    def __init__(self, cmodel: Model,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 writer: SummaryWriter,
                 similarity=torch.nn.CosineSimilarity(),
                 treshold: float = 0.95,
                 device: torch.device = torch.device('cpu'),
                 config=None,
                 model_forward=None
                 ):
        self.model = cmodel
        if model_forward:
            self.model_forward = model_forward
        else:
            self.model_forward = self.model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizers = []
        self.device = device
        self.similarity = similarity
        self.treshold = treshold
        self.writer = writer
        self.config = config
        if cmodel.optimizers and len(cmodel.optimizers) > 0:
            for index, optimizer in enumerate(cmodel.optimizers):
                if cmodel.optimizer_args and len(cmodel.optimizer_args) > index \
                        and cmodel.optimizer_args[index] != {}:
                    self.optimizers.append(optimizer(nn.ParameterList(self.model.parameters()),
                                                     **cmodel.optimizer_args[index]))
                else:
                    self.optimizers.append(optimizer(nn.ParameterList(self.model.parameters()),
                                                     **cmodel.default_optim_args))
            self.writer.add_hparams({'batch_size': self.train_loader.batch_size,
                                     'optimizers': str([opt.__class__.__name__ for opt in self.optimizers])},
                                    {})
        else:
            # no optimizer (including no lr scheduler), so const learning rate
            self.writer.add_hparams({'batch_size': self.train_loader.batch_size,
                                     'optimizer': 'None'},
                                    {'lr': self.model.lr})
        if self.model.lr_scheduler:
            self.model.lr_scheduler = self.model.lr_scheduler(self.optimizers[0],
                                                              **self.model.lr_scheduler_args)

    def train(self, epoch: int):
        self.model.if_train = True
        self.model.train()
        total_loss = 0
        correct = 0
        pbar = tqdm(total=len(self.train_loader),
                    desc="Training", dynamic_ncols=True)
        batch = 0
        for features, labels in self.train_loader:
            batch = batch + 1
            features = features.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            for index, optimizer in enumerate(self.optimizers):
                self.optimizers[index].zero_grad(set_to_none=True)
            outputs = self.model_forward(features)
            loss = self.model.loss(outputs, labels)
            loss.backward()
            if self.model.gradient_clipping:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.model.clip_value)
            for index, optimizer in enumerate(self.optimizers):
                if not self.model.closure or not self.model.closure[index]:
                    self.optimizers[index].step()
                else:
                    def closure_idx():
                        outputs = self.model(features)
                        loss = self.model.loss(outputs, labels)
                        loss.backward()
                        return loss

                    self.optimizers[index].step(closure_idx)
            if self.model.lr_scheduler:
                self.model.lr_scheduler.step()
            correct += (self.similarity(outputs, labels))
            total_loss += loss.item()
            pbar.set_postfix({'Loss': loss.item()})
            self.writer.add_scalar(f"Train/BLoss_{epoch}", loss.item(), batch)
            pbar.update()
        pbar.close()
        return total_loss, correct

    def val(self):
        self.model.if_train = False
        self.model.eval()
        correct = 0
        total_loss = 0
        with torch.no_grad():
            for features, labels in self.val_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model_forward(features)
                correct += (self.similarity(outputs, labels))
                total_loss += self.model.loss(outputs, labels).item()
        return total_loss, correct

    def run(self, n: int):
        metrics_memory = MetricsMemory(n)
        # timing_memory = MetricsMemory(n)
        start = timer()
        epochs_til_now = 0
        for epoch in range(n):
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizers[0].state_dict(),
            }, 'checkpoint')
            total_loss, correct_train = self.train(epoch)
            valid_loss, correct_test = self.val()
            train_metrics = (correct_train / (len(self.train_loader) * self.train_loader.batch_size),
                             total_loss / (len(self.train_loader) * self.train_loader.batch_size))
            val_metrics = (correct_test / (len(self.val_loader) * self.val_loader.batch_size),
                           valid_loss / (len(self.val_loader) * self.val_loader.batch_size))
            metrics_memory.update_metrics(epoch, val_metrics, train_metrics)

            self.writer.add_scalar("Train/Accuracy",
                                   correct_train / (len(self.train_loader) * self.train_loader.batch_size), epoch)
            self.writer.add_scalar("Val/Accuracy",
                                   correct_test / (len(self.val_loader) * self.val_loader.batch_size), epoch)
            self.writer.add_scalar("Train/Loss",
                                   total_loss / (len(self.train_loader) * self.train_loader.batch_size), epoch)
            self.writer.add_scalar("Val/Loss",
                                   valid_loss / (len(self.val_loader) * self.val_loader.batch_size), epoch)
            self.writer.add_scalar("Model/Norm", self.model.get_model_norm(), epoch)
            for opt in self.optimizers:
                self.writer.add_scalar(f'Model/Learning_Rate_{opt.__class__.__name__}',
                                       opt.param_groups[0]["lr"], epoch)
                # self.writer.flush()
            epochs_til_now = epoch
            if epoch > 10 and \
                    metrics_memory.metrics[epoch - 10][1] - metrics_memory.metrics[epoch][1] < 0.000001:
                break
            # accuracy = correct_test / (len(self.val_loader) * self.val_loader.batch_size)
            loss = valid_loss / (len(self.val_loader) * self.val_loader.batch_size)
            if self.config:
                with wandb.init(config=self.config):
                    wandb.log({"accuracy": loss, "epoch": epoch})
            # end = timer()
            # timing_memory.update_timing(epoch, end - start)
        end = timer()
        print("Elapsed time: ", end - start, " seconds")
        metrics_memory.draw_plot(epochs_til_now)
        # timing_memory.draw_timing_plot()
