import torch
import torch.nn.functional as F


class OneHot:
    def __init__(self, classes):
        self.classes = torch.tensor(classes, dtype=torch.long)

    def __call__(self, label):
        return (
            F.one_hot(torch.where(self.classes == label)[0], len(self.classes))
            .squeeze(0)
            .float()
        )
