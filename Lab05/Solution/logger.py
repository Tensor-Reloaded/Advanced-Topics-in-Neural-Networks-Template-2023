

from torch.utils.tensorboard import SummaryWriter
import wandb

class Logger:
    def __init__(
        self,
        tensorboard_file_suffix: str,
        log_to_wandb: bool = True
    ):
        self.tensorboard_writer = SummaryWriter(filename_suffix=tensorboard_file_suffix)
        self.log_to_wandb = log_to_wandb

    def log_scalar_for_batch(self, key: str, value: any, batch: int):
        self.tensorboard_writer.add_scalar(key, value, batch+1)

    def log_scalar_for_epoch(self, key: str, value: any, epoch: int):
        self.tensorboard_writer.add_scalar(key, value, epoch+1)
        if(self.log_to_wandb):      
            wandb.log({key: value}, step=epoch+1)
        
    def log_text_config(self, key: str, value: str):
        self.tensorboard_writer.add_text(key, value)
        if(self.log_to_wandb):
            wandb.config.update({key: value})

    def log_scalar_config(self, key: str, value: str):
        self.tensorboard_writer.add_scalar(key, value)
        if(self.log_to_wandb):
            wandb.config.update({key: value})