import os

import wandb


class WbaLogger:
    def __init__(self):
        secret = os.getenv("CLIENT_SECRET")
        wandb.login(secret)
        wandb.init(
            # set the wandb project where this run will be logged
            project="advanced_topics_lab6",

            # track hyperparameters and run metadata
            config={
                "learning_rate": 0.01,
                "architecture": "MLP",
                "dataset": "CIFAR-100",
                "epochs": 200,
                "batch_size": 512
            }
        )

    def info(self, data):
        wandb.log(data)
