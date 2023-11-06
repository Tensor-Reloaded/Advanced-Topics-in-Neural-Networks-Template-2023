import typing as t
import torch
import torch.utils.data as torch_data
from nn.optimisers import SAM
from nn.metered_trainable_model import MeteredTrainableNeuralNetwork


class MeteredTrainableSAMNeuralNetwork(MeteredTrainableNeuralNetwork):
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
        super(MeteredTrainableSAMNeuralNetwork, self).__init__(
            input_size=input_size,
            output_size=output_size,
            loss_function=loss_function,
            optimiser=lambda parameters, lr: SAM(
                parameters, optimiser, lr=lr
            ),
            learning_rate=learning_rate,
            output_layer_activation_function=output_layer_activation_function,
            device=device,
            log_directory=log_directory,
        )

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
            self.optimiser.first_step(zero_grad=True)

            y_hat = self(image)
            loss = self.loss_function(y_hat, label)
            loss.backward()
            self.optimiser.second_step(zero_grad=True)

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
