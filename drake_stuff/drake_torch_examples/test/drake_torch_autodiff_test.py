import unittest

import numpy as np
import torch

from pydrake.autodiffutils import (
    AutoDiffXd,
    ExtractGradient,
    ExtractValue,
    InitializeAutoDiff,
)
import pydrake.math as drake_math

from anzu.not_exported_soz.containers import take_first
from anzu.drake_torch_autodiff import (
    drake_torch_function,
)


def my_sin_torch(
    str_nograd, float_nograd, vector_nograd, matrix_grad, vector_grad
):
    assert str_nograd == "str_nograd"
    return (
        float_nograd * torch.sin(matrix_grad)
        + vector_grad**2
        + vector_nograd
    )


@np.vectorize
def drake_sin(x):
    return drake_math.sin(x)


def is_autodiff(x):
    assert x.size > 0
    if x.dtype != object:
        return False
    x0 = take_first(x.flat)
    return isinstance(x0, AutoDiffXd)


@drake_torch_function
def my_sin_drake(
    str_nograd, float_nograd, vector_nograd, matrix_grad, vector_grad
):
    assert str_nograd == "str_nograd"
    # Normally, you can write your Drake function to operation on
    # `T in [float, AutoDiffXd, Expression]`. We add these asserts just to
    # confirm that routing through `my_sin_drake.torch` converts all tensors,
    # regardless of whether they need gradients.
    assert is_autodiff(vector_nograd)
    assert is_autodiff(matrix_grad)
    assert is_autodiff(vector_grad)
    return (
        float_nograd * drake_sin(matrix_grad)
        + vector_grad**2
        + vector_nograd
    )


def my_sin_drake_batched(
    str_nograd, float_nograd, vector_nograd, matrix_grad, vector_grad
):
    """Example function that batches along (vector_nograd, matrix_grad)."""
    N = vector_nograd.shape[0]
    y = torch.zeros((N, 2, 2))
    for i in range(N):
        y[i] = my_sin_drake.torch(
            str_nograd,
            float_nograd,
            vector_nograd[i],
            matrix_grad[i],
            vector_grad,
        )
    return y


@drake_torch_function
def simple_drake(x):
    return x


class Test(unittest.TestCase):
    def calc_value_and_grad(self, func):
        str_nograd = "str_nograd"
        float_nograd = 2.0
        vector_nograd = torch.tensor([0.1])
        vector_grad = torch.tensor([10.0], requires_grad=True)
        # Add non-trivial (but simple) chain rule.
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        k = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
        matrix_grad = k * x
        matrix_grad = x
        # Compute value and backprop to gradient.
        y = func(
            str_nograd,
            float_nograd,
            vector_nograd,
            matrix_grad,
            vector_grad,
        )
        fake_loss = y.sum()
        fake_loss.backward()
        dL_dx = x.grad
        dL_dvector = vector_nograd.grad
        return fake_loss.detach(), dL_dx, dL_dvector

    def test_my_sin(self):
        L_torch, dL_dx_torch, dL_dvector_torch = self.calc_value_and_grad(
            my_sin_torch
        )
        L_drake, dL_dx_drake, dL_dvector_drake = self.calc_value_and_grad(
            my_sin_drake.torch
        )
        torch.testing.assert_close(L_torch, L_drake)
        torch.testing.assert_close(dL_dx_torch, dL_dx_drake)
        torch.testing.assert_close(dL_dvector_torch, dL_dvector_drake)

    def test_my_sin_batched(self):
        str_nograd = "str_nograd"
        float_nograd = 2.0
        vector_nograd = torch.tensor([[0.1], [0.2]])
        matrix_grad = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ],
            requires_grad=True,
        )
        vector_grad = torch.tensor([10.0], requires_grad=True)
        y = my_sin_drake_batched(
            str_nograd, float_nograd, vector_nograd, matrix_grad, vector_grad
        )
        fake_loss = y.sum()
        # Ensure we can backprop.
        fake_loss.backward()

    def test_cuda(self):
        device = torch.device("cuda")
        x = torch.tensor([1.0], requires_grad=True, device=device)
        y = simple_drake.torch(x)
        y.sum().backward()


if __name__ == "__main__":
    unittest.main()
