from torch.utils.tensorboard import SummaryWriter


class Config:
    def __init__(self, device, epochs, criterion, optimizer, learning_rate, legend, directory, base_optimizer=None, **kwargs):
        self.device = device
        self.epochs = epochs
        self.criterion = criterion
        self.optimizer = optimizer
        self.base_optimizer = base_optimizer
        self.additional_params = kwargs
        self.learning_rate = learning_rate
        self.legend = legend
        self.writer = SummaryWriter(directory)