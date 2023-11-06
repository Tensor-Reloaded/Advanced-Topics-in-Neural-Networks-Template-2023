

from torch.utils.tensorboard import SummaryWriter
import wandb

class Logger:
    def __init__(
        self,
        project_name: str
    ):
        self.tensorboard_writer = SummaryWriter(filename_suffix=project_name)
        self.wandb_run = wandb.init(project=project_name, mode="dryrun")

    def log_scalar_for_batch(self, key: str, value: any, batch: int):
        self.tensorboard_writer.add_scalar(key, value, batch+1)

    def log_scalar_for_epoch(self, key: str, value: any, epoch: int):
        self.tensorboard_writer.add_scalar(key, value, epoch+1)
        self.wandb_run.log({key: value}, step=epoch+1)
        
    def log_config(self, key: str, value: str):
        self.tensorboard_writer.add_text(key, value)
        self.wandb_run.config.update({key: value})