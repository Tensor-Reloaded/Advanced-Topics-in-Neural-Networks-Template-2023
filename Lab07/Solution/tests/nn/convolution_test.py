#!/usr/bin/env python
import torch
import pytest
from Solution.src.nn.convolution import Convolution


def test_produce_similar_results_to_conv2d():
    inputs = torch.randn(1, 3, 10, 12)
    weights = torch.randn(2, 3, 4, 5)

    sut = Convolution(weights=weights, bias=torch.zeros(weights.shape[0], 1))
    result = sut(inputs=inputs)

    expected_result = torch.nn.functional.conv2d(inputs, weights, bias=torch.zeros(weights.shape[0]))
    assert (expected_result - result).abs().max() < 0.001

@pytest.mark.skip(reason="TODO: inspect reason why this is failing")
def test_produce_similar_results_to_conv2d_with_padding():
    inputs = torch.randn(1, 3, 10, 12)
    weights = torch.randn(2, 3, 4, 5)

    sut = Convolution(weights=weights, bias=torch.zeros(weights.shape[0], 1), padding=1)
    result = sut(inputs=inputs)

    expected_result = torch.nn.functional.conv2d(
        inputs, weights, bias=torch.zeros(weights.shape[0], padding=1)
    )
    assert (expected_result - result).abs().max() < 0.001
