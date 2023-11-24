import torch
import wandb


class WBAManager:
    def __init__(self):
        WBAManager.login(self.get_api_key("api_key"))
        self.config = WBAManager.get_config()
        WBAManager.init_wandb(self.config)

    @staticmethod
    def get_api_key(path: str) -> str:
        with open(path, "r") as fd:
            return fd.read().replace("\n", "")

    @staticmethod
    def get_config() -> dict:
        data = {
            "name": "ConvMixer",

            "optimizer_name": "AdamW",

            "batch_size": 512,
            "scale":  0.75,
            "reprob":  0.25,
            "ra_m":  8,
            "ra_n":  1,
            "jitter":  0.1,

            "hdim": 256,
            "depth": 8,
            "psize": 2,
            "conv_ks": 5,

            "wd": 0.01,
            "clip_norm": False,
            "epochs": 25,
            "lr_max": 0.01,
            "workers": 2
        }

        data["lr_max"] = 0.05
        data["ra_n"] = 2
        data["ra_m"] = 12
        data["wd"] = 0.005
        data["scale"] = 1.0
        data["jitter"] = 0.2
        data["reprob"] = 0.2
        data["epochs"] = 150

        return data

    @staticmethod
    def init_wandb(config: dict):
        wandb.init(
            project="Tema7 Retele Neuronale",
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

