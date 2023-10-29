import typing as t
import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    __output_layer_activation_function: t.Callable[[torch.Tensor], torch.Tensor]

    def __init__(
        self,
        image_size: int,
        output_layer_activation_function=t.Union[
            None, t.Callable[[torch.Tensor], torch.Tensor]
        ],
    ) -> None:
        super(NeuralNetwork, self).__init__()

        self.__output_layer_activation_function = (
            output_layer_activation_function
            if output_layer_activation_function is not None
            else nn.Identity
        )

        self.layer_1_weights = nn.Linear(image_size, 1024)
        self.layer_2_weights = nn.Linear(1024, 64)
        self.layer_3_weights = nn.Linear(64, 1024)
        self.layer_4_weights = nn.Linear(1024, image_size)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        y_hat = torch.relu(self.fc1(X))
        y_hat = torch.relu(self.fc2(y_hat))
        y_hat = self.__output_layer_activation_function(self.fc3(y_hat))

        return y_hat
