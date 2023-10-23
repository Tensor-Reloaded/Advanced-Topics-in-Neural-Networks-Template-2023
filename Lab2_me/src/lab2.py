import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorboard


# Tensors are by default on CPU
# Please ensure that you have access to a GPU first (in Google Colab, change Runtime type to T4 GPU).
x = torch.arange(5, 15, 2)
x_cuda = x.to('cuda')