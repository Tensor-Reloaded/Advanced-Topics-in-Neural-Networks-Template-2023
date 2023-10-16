import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import Tensor
from typing import Tuple

print("da")
x = torch.rand(3, 784)
w = torch.rand(784, 10)
b = torch.rand(1, 10)
#fill with 10 random values of 0 and 1
y_result = torch.randint(0, 2, (1, 10))

print("Good Y vals:")
print(y_result)