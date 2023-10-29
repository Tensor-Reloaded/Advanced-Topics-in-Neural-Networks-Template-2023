import typing as t
import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    __output_layer_activation_function: t.Callable[[torch.Tensor], torch.Tensor]
    __device: str

    def __init__(
        self,
        image_size: int,
        output_layer_activation_function: t.Union[
            None, t.Callable[[torch.Tensor], torch.Tensor]
        ] = None,
        device: str = "cpu",
    ) -> None:
        super(NeuralNetwork, self).__init__()

        self.__output_layer_activation_function = (
            output_layer_activation_function
            if output_layer_activation_function is not None
            else nn.Identity()
        )
        self.__device = device

        self.layer_1_weights = nn.Linear(image_size, 1024).to(
            device=self.__device, non_blocking=self.__device == "cuda"
        )
        self.layer_1_normalisation = nn.BatchNorm1d(1024)
        self.layer_2_weights = nn.Linear(1024, 64).to(
            device=self.__device, non_blocking=self.__device == "cuda"
        )
        self.layer_2_normalisation = nn.BatchNorm1d(64)
        self.layer_3_weights = nn.Linear(64, 1024).to(
            device=self.__device, non_blocking=self.__device == "cuda"
        )
        self.layer_3_normalisation = nn.BatchNorm1d(1024)
        self.layer_4_weights = nn.Linear(1024, image_size).to(
            device=self.__device, non_blocking=self.__device == "cuda"
        )
        self.layer_4_normalisation = nn.BatchNorm1d(image_size)
        nn.init.kaiming_normal_(self.layer_1_weights.weight)
        nn.init.kaiming_normal_(self.layer_2_weights.weight)
        nn.init.kaiming_normal_(self.layer_3_weights.weight)
        nn.init.kaiming_normal_(self.layer_4_weights.weight)
        nn.init.zeros_(self.layer_1_weights.bias)
        nn.init.zeros_(self.layer_2_weights.bias)
        nn.init.zeros_(self.layer_3_weights.bias)
        nn.init.zeros_(self.layer_4_weights.bias)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        print(X.shape)
        y_hat = torch.relu(self.layer_1_normalisation(self.layer_1_weights(X)))
        y_hat = torch.relu(self.layer_2_normalisation(self.layer_2_weights(y_hat)))
        y_hat = torch.relu(self.layer_3_normalisation(self.layer_3_weights(y_hat)))
        y_hat = self.__output_layer_activation_function(
            self.layer_4_normalisation(self.layer_4_weights(y_hat))
        )

        return y_hat
