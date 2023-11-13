import torch
import wandb
from sam import SAM


class WBAManager:
    def __init__(self, optimizer_name: str, config_idx: int):
        WBAManager.login(self.get_api_key("api_key"))
        self.config = WBAManager.get_config(optimizer_name, config_idx)
        WBAManager.init_wandb(self.config)

    @staticmethod
    def get_api_key(path: str) -> str:
        with open(path, "r") as fd:
            return fd.read().replace("\n", "")

    @staticmethod
    def get_config(optimizer_name: str, config_idx: int) -> dict:
        custom_config = get_config(optimizer_name, config_idx)

        return {
            "architecture": "UpscaleNN",
            "dataset": "CIFAR-10",
            "val_batch_size": 500,
            **custom_config
        }

    @staticmethod
    def init_wandb(config: dict):
        wandb.init(
            project="Tema5 Retele Neuronale",
            config=config
        )

    @staticmethod
    def login(api_key: str):
        wandb.login(key=api_key)

    @staticmethod
    def log(data: dict):
        wandb.log(data)

    def __del__(self):
        wandb.finish()


CONFIG_0 = {
    "epochs": 50,
    "batch_size": 128,
    "learning_rate": 0.01,
    "momentum": 0.0,
    "decay": 0.0
}

CONFIG_1 = {
    "epochs": 40,
    "batch_size": 64,
    "learning_rate": 0.02,
    "momentum": 0.8,
    "decay": 0.0014
}

CONFIG_2 = {
    "epochs": 70,
    "batch_size": 128,
    "learning_rate": 0.005,
    "momentum": 0.4,
    "decay": 0.0024
}


def get_config(optimizer_name: str, config_idx: int) -> dict:
    configs: list[dict] = [CONFIG_0, CONFIG_1, CONFIG_2]

    if optimizer_name == "SGD":
        current_config = configs[config_idx].copy()
        current_config["optimizer_name"] = str(optimizer_name)
        current_config["get_optimizer_fn"] = lambda params, learning_rate=current_config["learning_rate"], decay=current_config["decay"], momentum=current_config["momentum"]: torch.optim.SGD(params, learning_rate, weight_decay=decay, momentum=momentum)
        return current_config

    if optimizer_name == "Adam":
        current_config = configs[config_idx].copy()
        current_config["optimizer_name"] = str(optimizer_name)
        current_config["get_optimizer_fn"] = lambda params, learning_rate=current_config["learning_rate"], decay=current_config["decay"], momentum=current_config["momentum"]: torch.optim.Adam(params, learning_rate, weight_decay=decay)
        return current_config

    if optimizer_name == "RMSprop":
        current_config = configs[config_idx].copy()
        current_config["optimizer_name"] = str(optimizer_name)
        current_config["get_optimizer_fn"] = lambda params, learning_rate=current_config["learning_rate"], decay=current_config["decay"], momentum=current_config["momentum"]: torch.optim.RMSprop(params, learning_rate, weight_decay=decay, momentum=momentum)
        return current_config

    if optimizer_name == "Adagrad":
        current_config = configs[config_idx].copy()
        current_config["optimizer_name"] = str(optimizer_name)
        current_config["get_optimizer_fn"] = lambda params, learning_rate=current_config["learning_rate"], decay=current_config["decay"], momentum=current_config["momentum"]: torch.optim.Adagrad(params, learning_rate, weight_decay=decay)
        return current_config

    if optimizer_name == "SGD_SAM":
        current_config = configs[config_idx].copy()
        current_config["optimizer_name"] = str(optimizer_name)

        current_config["get_optimizer_fn"] = lambda params, learning_rate=current_config["learning_rate"], decay=current_config["decay"], momentum=current_config["momentum"]: SAM(params, torch.optim.SGD, lr=learning_rate,  weight_decay=decay, momentum=momentum)
        return current_config

    assert False
