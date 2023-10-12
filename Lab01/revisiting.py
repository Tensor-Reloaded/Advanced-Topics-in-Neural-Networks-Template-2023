from typing import Tuple

import torch
from torch import Tensor


def print_tensor(name: str, x: Tensor):
    print(name)
    print(x)
    print()


def setup_col():
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


def setup_row():
    x = Tensor([[1.0], [3.0], [0.0]]).T
    y = Tensor([[0], [1], [0]]).T
    w = Tensor([
        [0.3, 0.1, -2.0],
        [-0.6, -0.5, 2.0],
        [-1.0, -0.5, 0.1]
    ])
    b = Tensor([[0.1], [0.1], [0.1]]).T
    lr = 0.2
    assert x.shape == (1, 3)
    assert y.shape == (1, 3)
    assert b.shape == (1, 3)
    assert w.shape == (3, 3), "w must be a 3x3 matrix"
    print_tensor("x", x)
    print_tensor("y", y)
    print_tensor("w", w)
    print_tensor("b", b)
    return x, y, w, b, lr


def forward_col(x: Tensor, w: Tensor, b: Tensor) -> Tensor:
    return w.T @ x + b


def forward_row(x: Tensor, w: Tensor, b: Tensor) -> Tensor:
    return x @ w + b


def activate_col(x: Tensor) -> Tensor:
    return x.softmax(dim=0)


def activate_row(x: Tensor) -> Tensor:
    return x.softmax(dim=1)


def backward_col(x: Tensor, y: Tensor, y_hat: Tensor) -> Tuple[Tensor, Tensor]:
    error = y_hat - y
    delta_w = error @ x.T
    delta_w_t = delta_w.T
    delta_b = error
    return delta_w_t, delta_b


def backward_row(x: Tensor, y: Tensor, y_hat: Tensor) -> Tuple[Tensor, Tensor]:
    error = y_hat - y
    delta_w = x.T @ error
    delta_b = error.mean(dim=0)  # On column
    return delta_w, delta_b


def raw_pytorch_col():
    print("Raw Pytorch")
    x, y, w, b, lr = setup_col()
    z = forward_col(x, w, b)
    y_hat = activate_col(z)
    delta_w_t, delta_b = backward_col(x, y, y_hat)
    print_tensor("delta_w_t", delta_w_t)
    w -= lr * delta_w_t
    b -= lr * delta_b
    print_tensor("w", w)
    print_tensor("b", b)
    return w, b


def raw_pytorch_row():
    print("Raw Pytorch")
    x, y, w, b, lr = setup_row()
    z = forward_row(x, w, b)
    y_hat = activate_row(z)
    delta_w_t, delta_b = backward_row(x, y, y_hat)
    print_tensor("delta_w_t", delta_w_t)
    w -= lr * delta_w_t
    b -= lr * delta_b
    print_tensor("w", w)
    print_tensor("b", b)
    return w, b


def pytorch_with_autograd_col():
    print("Pytorch with Autograd")
    x, y, w, b, lr = setup_col()
    w.requires_grad_()  # inplace operation
    b.requires_grad_()
    z = forward_col(x, w, b)
    loss = torch.nn.functional.cross_entropy(z.unsqueeze(0), y.unsqueeze(0))
    loss.backward()
    print_tensor("w.grad", w.grad)
    w = w - lr * w.grad
    b = b - lr * b.grad
    print_tensor("w", w)
    print_tensor("b", b)
    return w, b


def pytorch_with_autograd_row():
    print("Pytorch with Autograd")
    x, y, w, b, lr = setup_row()
    w.requires_grad_()  # inplace operation
    b.requires_grad_()
    z = forward_row(x, w, b)
    loss = torch.nn.functional.cross_entropy(z, y)
    loss.backward()
    print_tensor("w.grad", w.grad)
    w = w - lr * w.grad
    b = b - lr * b.grad
    print_tensor("w", w)
    print_tensor("b", b)
    return w, b


def main():
    w1, b1 = raw_pytorch_col()
    w2, b2 = pytorch_with_autograd_col()
    assert torch.allclose(w1, w2)
    assert torch.allclose(b1, b2)
    w3, b3 = raw_pytorch_row()
    w4, b4 = pytorch_with_autograd_row()
    assert torch.allclose(w3, w4)
    assert torch.allclose(b3, b3)

    assert torch.allclose(w1, w3)
    assert torch.allclose(b1, b3.T)


if __name__ == '__main__':
    main()
