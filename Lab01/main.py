import torch
import torch.nn.functional as F
import numpy as np

# Learning rate
learning_rate = 0.1

# Weights
w = torch.tensor([
    [0.3, 0.1, -2.0],
    [-0.6, -0.5, 2.0],
    [-1.0, -0.5, 0.1]
])
# Biases
b = torch.tensor([[0.1], [0.1], [0.1]])
# Labels
y = torch.tensor([[0], [1], [0]])
x = torch.tensor([[1.0], [3.0], [0.0]])

# Linear combinations
transpose_w = torch.transpose(w, 0, 1)
z = torch.matmul(transpose_w, x) + b
print(f"Linear combinations:\n{z.numpy()}")

# Predicted probabilities
predicted_probabilities = F.softmax(z, dim=0)
print(f"Predicted probabilities:\n{predicted_probabilities.numpy()}")

# Calculate gradients for weights and biases
db_l = predicted_probabilities - y
print(f"Biases gradient:\n{db_l.numpy()}")
dw_l = torch.mul(db_l.t(), x)
print(f"Weights gradient:\n{dw_l.numpy()}")

b = b - learning_rate * db_l
w = w - learning_rate * dw_l

print(f"Updated biases:\n{b.numpy()}")
print(f"Updated weights:\n{w.numpy()}")