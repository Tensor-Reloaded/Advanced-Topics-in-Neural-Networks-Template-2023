from math import pow, e
from typing import List

Vector = List[float]

def main():
    x = [1, 3, 0]
    w = [-0.6, -0.5, 2]
    b = [0.1, 0.1, 0.1]
    y = 1

    w_prime, b_prime = iterate(x, w, b, y)

    print(w_prime)
    print(b_prime)

def iterate(w: Vector, x: Vector, b: Vector, y: float) -> Vector:
    y_prime = sigmoid(sum([weight * value for weight, value in zip(w, x)]) + sum(b))

    error = y - y_prime

    delta_w = [-error * weight for weight in w]
    delta_b = [-error for _ in b]

    w_prime = [weight - delta_weight for weight, delta_weight in zip(w, delta_w)]
    b_prime = [bias - delta_bias for bias, delta_bias in zip(b, delta_b)]

    return w_prime, b_prime

def sigmoid(x: float) -> float:
    return 1 / (1 + e ** -x)

if __name__ == '__main__':
    main()
