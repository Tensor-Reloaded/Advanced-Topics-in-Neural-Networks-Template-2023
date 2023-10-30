import typing as t
from functools import reduce
import torch
import torch.utils.data as torch_data
from nn import NeuralNetwork


def run(
    train: t.Callable[[], t.List[torch.Tensor]],
    val: t.Callable[[], t.List[torch.Tensor]],
    epochs: int,
) -> t.Tuple[t.List[t.Tuple[int, float]], t.List[t.Tuple[int, float]]]:
    training_loss_means: t.List[t.Tuple[int, float]] = []
    validation_loss_means: t.List[t.Tuple[int, float]] = []

    try:
        for epoch in range(0, epochs):
            training_losses = train()
            validation_losses = val()

            training_loss_mean = (
                reduce(lambda x, y: x + y, training_losses) / len(training_losses)
            ).item()
            validation_loss_mean = (
                reduce(lambda x, y: x + y, validation_losses) / len(validation_losses)
            ).item()
            training_loss_means.append((epoch, training_loss_mean))
            validation_loss_means.append((epoch, validation_loss_mean))

            print(
                f"Training epoch {epoch + 1}: training loss mean = {training_loss_mean}, validation loss mean = {validation_loss_mean}",
                end="\r",
            )

    except StopIteration:
        pass

    print()

    return training_loss_means, validation_loss_means


def train(
    model: NeuralNetwork,
    optimiser: torch.optim.SGD,
    loss_function: torch.nn.MSELoss,
    train_dataloader: torch_data.DataLoader,
):
    def fn() -> t.List[torch.Tensor]:
        model.train()
        training_losses: t.List[torch.Tensor] = []

        for training_image_set in train_dataloader:
            start_image = training_image_set[0]
            end_image = training_image_set[1]
            time_skip = training_image_set[2]

            optimiser.zero_grad()
            y_hat = model(start_image, time_skip)
            loss = loss_function(y_hat, end_image)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimiser.step()

            training_losses.append(loss)

        return training_losses

    return fn


def val(
    model: NeuralNetwork,
    loss_function: torch.nn.MSELoss,
    validation_dataloader: torch_data.DataLoader,
):
    def fn() -> t.List[torch.Tensor]:
        model.eval()
        validation_losses: t.List[torch.Tensor] = []

        for validation_image_set in validation_dataloader:
            start_image = validation_image_set[0]
            end_image = validation_image_set[1]
            time_skip = validation_image_set[2]

            y_hat = model(start_image, time_skip)
            loss = loss_function(y_hat, end_image)

            validation_losses.append(loss)

        return validation_losses

    return fn


def test(model: NeuralNetwork, test_dataloader: torch_data.DataLoader):
    model.eval()
    total = 0
    correct = 0
    accuracy_threshold = 0.2

    with torch.no_grad():
        for test_image_set in test_dataloader:
            start_image = test_image_set[0]
            end_image = test_image_set[1]
            time_skip = test_image_set[2]

            y_hat = model(start_image, time_skip)
            total += end_image.size(0)
            correct += ((y_hat - end_image).abs() < accuracy_threshold).sum().item()

            print(f"Testing: Accuracy = {correct / total:.2f}%", end="\r")

    print()
