from torch.utils.tensorboard import SummaryWriter

class Config:
    def __init__(self, device, epochs, criterion, optimizer, learning_rate, legend, dir, base_optimizer=None):
        self.device = device
        self.epochs = epochs
        self.criterion = criterion
        self.optimizer = optimizer
        self.base_optimizer = base_optimizer
        self.learning_rate = learning_rate
        self.legend = legend
        self.writer= SummaryWriter(dir)