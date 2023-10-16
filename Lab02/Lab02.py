from typing import Tuple

import torch
from torch import Tensor


def train_perceptron(X: Tensor, W: Tensor, b: Tensor, y_true: Tensor, mu: float) -> Tuple[Tensor, Tensor]:
    z = X @ W + b
    y = z.sigmoid()
    error = y_true - y
    updated_weight = W + mu * error @ X
    updated_bias = b + mu * error
    return updated_weight, updated_bias

def main():
    m = int(input("Enter the inputs count "))
    mu = float(input("Enter the learning rate "))
    num_matrix = list()
    for i in range(int(m)):
        row = list()
        for j in range(10):
            print("Enter the y_true element on position ", i, " ", j)
            n = int(input())
            row.append(int(n))
        num_matrix.append(row)
    x = torch.rand(m, 784)
    w = torch.rand(784, 10)
    b = torch.rand(10, 1)
    y_true = Tensor(num_matrix)

    output_tuple = train_perceptron(x, w, b, y_true, mu)
    print("Updated Weight")
    print(output_tuple[0])
    print()
    print("Updated Bias")
    print(output_tuple[1])

main()
