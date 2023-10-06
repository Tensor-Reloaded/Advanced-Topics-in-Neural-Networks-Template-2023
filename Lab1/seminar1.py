#!/usr/bin/env python3
import numpy as np
import math
import numpy.typing as npt

Vector = npt.NDArray[np.float32]

def main():
    x = np.array([1, 3, 0], dtype=np.float32)
    w = np.array([-0.6, -0.5, 2], dtype=np.float32)
    b = np.array([0.1, 0.1, 0.1], dtype=np.float32)
    y = np.float32(1)

    w_prime, b_prime = iterate(x, w, b, y)

    print(w_prime)
    print(b_prime)

def iterate(x: Vector, w: Vector, b: Vector, y: np.float32) -> Vector:
    wx_plus_b = np.sum(w * x) + np.sum(b)
    y_prime = activation_function(wx_plus_b)

    error = loss_function(y, y_prime)

    delta_w = -error * w
    delta_b = -error * np.ones_like(b)

    w_prime = w - delta_w
    b_prime = b - delta_b

    return w_prime, b_prime

def activation_function(x: np.float32) -> np.float32:
    return 1 / (1 + math.e ** -x)

def loss_function(y: np.float32, y_prime: np.float32) -> np.float32:
    return y - y_prime

if __name__ == '__main__':
    main()
