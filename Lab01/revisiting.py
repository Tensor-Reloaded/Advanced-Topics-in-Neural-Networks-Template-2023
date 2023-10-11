from typing import Tuple

import torch
from torch import Tensor


def print_tensor(name: str, x: Tensor):
    print(name)
    print(x)
    print()


def setup():
    x = Tensor([[1.0], [3.0], [0.0]])
    y = Tensor([[0], [1], [0]])
    w = Tensor([
        [0.3, 0.1, -2.0],
        [-0.6, -0.5, 2.0],
        [-1.0, -0.5, 0.1]
    ])
    b = Tensor([[0.1], [0.1], [0.1]])
    lr = 0.2
    assert x.shape == (3, 1), "x must be a column vector"
    assert y.shape == (3, 1), "y must be a column vector"
    assert b.shape == (3, 1), "b must be a column vector"
    assert w.shape == (3, 3), "w must be a 3x3 matrix"
    print_tensor("x", x)
    print_tensor("y", y)
    print_tensor("w", w)
    print_tensor("b", b)
    return x, y, w, b, lr


def softmax(x: Tensor) -> Tensor:
    exp = x.exp()
    return exp / exp.sum(dim=0)


def forward(x: Tensor, w: Tensor, b: Tensor) -> Tensor:
    return w.T @ x + b


def activate(x: Tensor) -> Tensor:
    return softmax(x)


def backward(x: Tensor, y: Tensor, y_hat: Tensor) -> Tuple[Tensor, Tensor]:
    error = y_hat - y
    delta_w = error @ x.T
    delta_w_t = delta_w.T
    delta_b = error
    return delta_w_t, delta_b


def raw_pytorch():
    print("Raw Pytorch")
    x, y, w, b, lr = setup()
    z = forward(x, w, b)
    y_hat = activate(z)
    delta_w_t, delta_b = backward(x, y, y_hat)
    print_tensor("delta_w_t", delta_w_t)
    w -= lr * delta_w_t
    b -= lr * delta_b
    print_tensor("w", w)
    print_tensor("b", b)
    return w, b


def pytorch_with_autograd():
    print("Pytorch with Autograd")
    x, y, w, b, lr = setup()
    w.requires_grad_()  # inplace operation
    b.requires_grad_()
    z = forward(x, w, b)
    loss = torch.nn.functional.cross_entropy(z.unsqueeze(0), y.unsqueeze(0))
    loss.backward()
    print_tensor("w.grad", w.grad)
    w = w - lr * w.grad
    b = b - lr * b.grad
    print_tensor("w", w)
    print_tensor("b", b)
    return w, b


def main():
    w1, b1 = raw_pytorch()
    w2, b2 = pytorch_with_autograd()
    assert torch.allclose(w1, w2)
    assert torch.allclose(b1, b2)


if __name__ == '__main__':
    main()
