import typing as t
import torch


class Convolution(torch.nn.Module):
    __weights: torch.Tensor
    __bias: torch.Tensor
    __padding: int

    def __init__(
        self,
        weights: torch.Tensor,
        bias: t.Union[None, torch.Tensor] = None,
        padding: t.Union[None, int] = None,
    ) -> None:
        super(Convolution, self).__init__()
        self.__weights = weights
        self.__bias = (
            bias if bias is not None else torch.zeros(self.__weights.shape[0], 1)
        )
        self.__padding = padding if padding is not None else 0

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, _, in_height, in_width = inputs.size()
        out_channels, _, kernel_height, kernel_width = self.__weights.size()

        out_height = in_height - kernel_height + 1
        out_width = in_width - kernel_width + 1

        output = torch.zeros(
            batch_size,
            out_channels,
            out_height + self.__padding * 2,
            out_width + self.__padding * 2,
        )

        inputs_padded = inputs

        if self.__padding != 0:
            inputs_padded = torch.zeros(
                inputs.shape[0],
                inputs.shape[1],
                inputs.shape[2] + self.__padding * 2,
                inputs.shape[3] + self.__padding * 2,
            )
            inputs_padded[
                :,
                :,
                self.__padding : -1 * self.__padding,
                self.__padding : -1 * self.__padding,
            ] = inputs

        for b in range(batch_size):
            for o in range(out_channels):
                for i in range(out_height):
                    for j in range(out_width):
                        output[b, o, i, j] = (
                            inputs_padded[
                                b, :, i : i + kernel_height, j : j + kernel_width
                            ]
                            * self.__weights[o]
                        ).sum() + self.__bias[o]

        return output
