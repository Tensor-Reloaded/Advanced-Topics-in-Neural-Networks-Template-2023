import torch
import time
import typing as t
from torch.nn import Module
import torch.utils.data as torch_data
from nn.estimators.base_accuracy_estimator import BaseAccuracyEstimator


class ModelTrainer:
    __model: Module
    __loss_function: torch.nn.modules.loss._Loss
    __optimiser: torch.optim.Optimizer
    __training_accuracy_estimator: BaseAccuracyEstimator
    __validation_accuracy_estimator: BaseAccuracyEstimator
    __device: torch.device
    __exports_path: str

    def __init__(
        self,
        model: torch.nn.Module,
        loss_function: torch.nn.modules.loss._Loss,
        optimiser: torch.optim.Optimizer,
        learning_rate: float,
        accuracy_estimator: BaseAccuracyEstimator,
        device: torch.device = torch.device("cpu"),
        exports_path: str = "/tmp",
    ) -> None:
        self.__model = model
        self.__loss_function = loss_function()
        self.__optimiser = optimiser(self.__model.parameters(), lr=learning_rate)
        self.__training_accuracy_estimator = accuracy_estimator
        self.__validation_accuracy_estimator = accuracy_estimator
        self.__device = device
        self.__exports_path = exports_path

        self.__loss_function = self.__loss_function.to(
            device=self.__device, non_blocking=self.__device == "cuda"
        )

    def run(
        self,
        batched_training_dataset: torch_data.DataLoader,
        batched_validation_dataset: torch_data.DataLoader,
        epochs: int,
    ):
        epochs_digits = len(str(epochs))

        for epoch in range(0, epochs):
            training_loss, training_accuracy = self.run_training(
                batched_training_dataset
            )
            validation_loss, validation_accuracy = self.run_validation(
                batched_validation_dataset
            )

            print(
                f"Training epoch {epoch + 1:>{epochs_digits}}: training loss = {training_loss:>8.2f}, training accuracy = {training_accuracy * 100:>6.2f}%, validation loss = {validation_loss:>8.2f}, validation accuracy = {validation_accuracy * 100:>6.2f}%",
                end="\r",
            )

        print()

    def run_training(
        self,
        batched_training_dataset: torch_data.DataLoader,
    ) -> t.Tuple[float, float]:
        self.__model.train()
        training_loss = 0.0

        for training_image in batched_training_dataset:
            image, label = training_image
            image = image.to(device=self.__device, non_blocking=self.__device == "cuda")
            label = label.to(device=self.__device, non_blocking=self.__device == "cuda")

            self.__optimiser.zero_grad()
            y_hat = self.__model(x=image)
            loss = self.__loss_function(y_hat, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                parameters=self.__model.parameters(), max_norm=1.0
            )
            self.__optimiser.step()

            self.__training_accuracy_estimator.update(y_hat=y_hat, y=label)
            training_loss += loss.item()

        return training_loss, self.__training_accuracy_estimator.get()

    def run_validation(
        self,
        batched_validation_dataset: torch_data.DataLoader,
    ) -> t.Tuple[float, float]:
        self.__model.eval()
        validation_loss = 0.0

        with torch.no_grad():
            for validation_image in batched_validation_dataset:
                image, label = validation_image
                image = image.to(
                    device=self.__device, non_blocking=self.__device == "cuda"
                )
                label = label.to(
                    device=self.__device, non_blocking=self.__device == "cuda"
                )

                y_hat = self.__model(image)
                loss = self.__loss_function(y_hat, label)

                self.__validation_accuracy_estimator.update(y_hat=y_hat, y=label)
                validation_loss += loss.item()

        return validation_loss, self.__validation_accuracy_estimator.get()

    def export(self) -> None:
        torch.save(
            self.__model.state_dict(), f"{self.__exports_path}/{time.time_ns()}.pt"
        )
