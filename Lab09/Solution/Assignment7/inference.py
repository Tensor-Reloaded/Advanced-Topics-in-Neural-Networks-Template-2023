import torch
from model import ImageTransformCNN


def load_model(path):
    state_dict = torch.load(path)
    model = ImageTransformCNN()
    model.load_state_dict(state_dict)
    return model
