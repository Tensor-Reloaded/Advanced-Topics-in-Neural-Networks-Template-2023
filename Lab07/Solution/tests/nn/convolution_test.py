#!/usr/bin/env python
import torch
from Solution.src.nn.convolution import Convolution


def test_should_produce_a_good_result():
    inputs = torch.randn(1, 3, 10, 12)
    weights = torch.randn(2, 3, 4, 5)

    custom_conv2d_layer = Convolution(weights=weights)
    out = custom_conv2d_layer(inputs=inputs)

    assert torch.nn.functional.conv2d(inputs, weights) == out
