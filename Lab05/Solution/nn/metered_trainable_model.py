import typing as t
import torch
import torch.utils.data as torch_data
from nn.trainable_model import TrainableNeuralNetwork
from torch.utils.tensorboard import SummaryWriter


class MeteredTrainableNeuralNetwork(TrainableNeuralNetwork):
    summary_writer: SummaryWriter

    def __init__(
        self,
        input_size: int,
        output_size: int,
        loss_function: torch.nn.modules.loss._Loss,
        optimiser: torch.optim.Optimizer,
        learning_rate: float,
        output_layer_activation_function: t.Union[
            t.Callable[[torch.Tensor], torch.Tensor], None
        ] = None,
        device: str = "cpu",
        log_directory: str = "../data",
    ) -> None:
        super(MeteredTrainableNeuralNetwork, self).__init__(
            input_size=input_size,
            output_size=output_size,
            loss_function=loss_function,
            optimiser=optimiser,
            learning_rate=learning_rate,
            output_layer_activation_function=output_layer_activation_function,
            device=device,
        )

        self.summary_writer = SummaryWriter(log_dir=log_directory)

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

            self.summary_writer.add_scalar("Training loss/epoch", training_loss, epoch)
            self.summary_writer.add_scalar(
                "Training accuracy/epoch", training_accuracy, epoch
            )
            self.summary_writer.add_scalar(
                "Validation loss/epoch", validation_loss, epoch
            )
            self.summary_writer.add_scalar(
                "Validation accuracy/epoch", validation_accuracy, epoch
            )
            self.summary_writer.add_scalar("Norm/epoch", self.get_norm(), epoch)
            self.summary_writer.add_scalar(
                "Learning rate/epoch", self.optimiser.param_groups[0]["lr"], epoch
            )
            self.summary_writer.add_scalar(
                "Batch size", next(iter(batched_training_dataset))[0].shape[0]
            )
            self.summary_writer.add_text(
                "Optimiser",
                str(self.optimiser),
            )

            print(
                f"Training epoch {epoch + 1:>{epochs_digits}}: training loss = {training_loss:>8.2f}, training accuracy = {training_accuracy * 100:>6.2f}%, validation loss = {validation_loss:>8.2f}, validation accuracy = {validation_accuracy * 100:>6.2f}%",
                end="\r",
            )

        print()
        self.summary_writer.flush()

    def run_training(
        self,
        batched_training_dataset: torch_data.DataLoader,
    ) -> t.Tuple[float, float]:
        self.train()
        batch = 0
        total = 0
        correct = 0
        training_loss = 0.0

        for training_image in batched_training_dataset:
            image, label = training_image
            image = image.to(device=self.device, non_blocking=self.device == "cuda")
            label = label.to(device=self.device, non_blocking=self.device == "cuda")

            self.optimiser.zero_grad()
            y_hat = self(image)
            loss = self.loss_function(y_hat, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimiser.step()

            self.summary_writer.add_scalar("Training loss/batch", loss, batch)

            for i in range(label.shape[0]):
                correct += (
                    torch.argmax(y_hat[i]).item() == torch.argmax(label[i]).item()
                )
                total += 1
            training_loss += loss.item()
            batch += 1

        accuracy = correct / total
        return training_loss, accuracy

    def run_validation(
        self,
        batched_validation_dataset: torch_data.DataLoader,
    ) -> t.Tuple[float, float]:
        self.eval()
        batch = 0
        total = 0
        correct = 0
        validation_loss = 0.0

        with torch.no_grad():
            for validation_image in batched_validation_dataset:
                image, label = validation_image
                image = image.to(device=self.device, non_blocking=self.device == "cuda")
                label = label.to(device=self.device, non_blocking=self.device == "cuda")

                y_hat = self(image)
                loss = self.loss_function(y_hat, label)

                for i in range(label.shape[0]):
                    correct += (
                        torch.argmax(y_hat[i]).item() == torch.argmax(label[i]).item()
                    )
                    total += 1

                self.summary_writer.add_scalar("Validation loss/batch", loss, batch)

                validation_loss += loss.item()
                batch += 1

        accuracy = correct / total
        return validation_loss, accuracy
