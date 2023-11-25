import torch


class Handmade_conv2d_implementation:
    def __init__(self, weights):
        self.weights = weights

    def __call__(self, input):
        # forward
        var = torch.nn.functional.conv2d(input, self.weights)
        new_input = torch.zeros(torch.Size([input.shape[0], self.weights.shape[0],
                                            input.shape[2] - self.weights.shape[2] + 1,
                                            input.shape[3] - self.weights.shape[3] + 1]))
        for lin in range(input.shape[2] - self.weights.shape[2] + 1):
            for col in range(input.shape[3] - self.weights.shape[3] + 1):
                for channel in range(self.weights.shape[0]):
                    # top left corner of filter
                    new_input[:, channel, lin, col] = (input[:, :, lin:lin + self.weights.shape[2],
                                                       col:col + self.weights.shape[3]]
                                                       * self.weights[channel]).sum(dim=(2, 3)).sum(dim=1)
        return new_input
