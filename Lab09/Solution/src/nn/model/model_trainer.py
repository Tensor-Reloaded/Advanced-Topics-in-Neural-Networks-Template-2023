import torch
import time
import typing as t
from torch.nn import Module
import torch.utils.data as torch_data
from nn.estimators.base_accuracy_estimator import BaseAccuracyEstimator


class ModelTrainer:
    _model: Module
    _loss_function: torch.nn.modules.loss._Loss
    _optimiser: torch.optim.Optimizer
    _training_accuracy_estimator: BaseAccuracyEstimator
    _validation_accuracy_estimator: BaseAccuracyEstimator
    _device: torch.device
    _exports_path: str

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
        self._model = model
        self._loss_function = loss_function()
        self._optimiser = optimiser(self._model.parameters(), lr=learning_rate)
        self._training_accuracy_estimator = accuracy_estimator
        self._validation_accuracy_estimator = accuracy_estimator
        self._device = device
        self._exports_path = exports_path

        self._loss_function = self._loss_function.to(
            device=self._device, non_blocking=self._device == "cuda"
        )

    def run(
        self,
        batched_training_dataset: torch_data.DataLoader,
        batched_validation_dataset: torch_data.DataLoader,
        epochs: int,
        training_patience: int = 5,
    ):
        training_patience_counter = 0
        best_validation_loss = None
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

            if best_validation_loss is None or best_validation_loss > validation_loss:
                best_validation_loss = validation_loss
                training_patience_counter = 0
            else:
                training_patience_counter += 1

                if training_patience_counter > training_patience:
                    print()
                    print(
                        f"Early stopping after {training_patience_counter} epochs with no improvement"
                    )
                    break

        print()

    def run_training(
        self,
        batched_training_dataset: torch_data.DataLoader,
    ) -> t.Tuple[float, float]:
        self._model.train()
        training_loss = 0.0

        for training_image in batched_training_dataset:
            image, label = training_image
            image = image.to(device=self._device, non_blocking=self._device == "cuda")
            label = label.to(device=self._device, non_blocking=self._device == "cuda")

            self._optimiser.zero_grad()
            y_hat = self._model(x=image)
            loss = self._loss_function(y_hat, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                parameters=self._model.parameters(), max_norm=1.0
            )
            self._optimiser.step()

            self._training_accuracy_estimator.update(y_hat=y_hat, y=label)
            training_loss += loss.item()

        return training_loss, self._training_accuracy_estimator.get()

    def run_validation(
        self,
        batched_validation_dataset: torch_data.DataLoader,
    ) -> t.Tuple[float, float]:
        self._model.eval()
        validation_loss = 0.0

        with torch.no_grad():
            for validation_image in batched_validation_dataset:
                image, label = validation_image
                image = image.to(
                    device=self._device, non_blocking=self._device == "cuda"
                )
                label = label.to(
                    device=self._device, non_blocking=self._device == "cuda"
                )

                y_hat = self._model(image)
                loss = self._loss_function(y_hat, label)

                self._validation_accuracy_estimator.update(y_hat=y_hat, y=label)
                validation_loss += loss.item()

        return validation_loss, self._validation_accuracy_estimator.get()

    def export(self) -> None:
        torch.save(
            self._model.state_dict(), f"{self._exports_path}/{time.time_ns()}.pt"
        )
