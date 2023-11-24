import torch


class Utils:
    @staticmethod
    def initialize_device(device=None):
        if device is not None:
            return torch.device(device)
        else:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")