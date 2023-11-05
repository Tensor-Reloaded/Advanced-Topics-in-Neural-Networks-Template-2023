import typing as t
import torch
import torch.utils.data as torch_data
from nn.model import NeuralNetwork


class TrainableNeuralNetwork(NeuralNetwork):
    loss_function: torch.nn.modules.loss._Loss
    optimiser: torch.optim.Optimizer

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
    ) -> None:
        super(TrainableNeuralNetwork, self).__init__(
            input_size=input_size,
            output_size=output_size,
            output_layer_activation_function=output_layer_activation_function,
            device=device,
        )

        self.loss_function = loss_function()
        self.optimiser = optimiser(self.parameters(), lr=learning_rate)

    def run(
        self,
        batched_training_dataset: torch_data.DataLoader,
        batched_validation_dataset: torch_data.DataLoader,
        epochs: int,
    ):
        for epoch in range(0, epochs):
            training_loss = self.run_training(batched_training_dataset)
            validation_loss, accuracy = self.run_validation(batched_validation_dataset)

            print(
                f"Training epoch {epoch + 1}: training loss = {training_loss:>8.2f}, validation loss = {validation_loss:>8.2f}, validation accuracy = {accuracy * 100:>6.2f}%",
                end="\r",
            )

        print()

    def run_training(
        self,
        batched_training_dataset: torch_data.DataLoader,
    ) -> float:
        self.train()
        training_loss = 0.0

        for training_image in batched_training_dataset:
            image, label = training_image
            image = image.to(self.device)
            label = label.to(self.device)

            self.optimiser.zero_grad()
            y_hat = self(image)
            loss = self.loss_function(y_hat, label)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

            self.optimiser.step()

            training_loss += loss.item()

        return training_loss

    def run_validation(
        self,
        batched_validation_dataset: torch_data.DataLoader,
    ) -> t.Tuple[float, float]:
        self.eval()
        total = 0
        correct = 0
        validation_loss = 0.0

        with torch.no_grad():
            for validation_image in batched_validation_dataset:
                image, label = validation_image
                image = image.to(self.device)
                label = label.to(self.device)

                y_hat = self(image)
                loss = self.loss_function(y_hat, label)

                for i in range(label.shape[0]):
                    correct += (
                        torch.argmax(y_hat[i]).item() == torch.argmax(label[i]).item()
                    )
                    total += 1

                validation_loss += loss.item()

        accuracy = correct / total

        return validation_loss, accuracy
