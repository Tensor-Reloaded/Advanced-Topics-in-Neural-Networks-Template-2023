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
        return {
            "learning_rate": 0.005,
            "architecture": "CompressionNN",
            "dataset": "CIFAR-10",
            "epochs": 200,
            "batch_size": 64,
            "val_batch_size": 500
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
