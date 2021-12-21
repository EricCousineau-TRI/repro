"""
Transforms a simple network to use *very* basic forward auto diff.

Of course, this suffers when matrix dim grows too large.
"""

import torch
from torch import nn


def torch_col_zero(A, mask):
    # A: N x R x C
    # mask: N x C
    N, R, C = A.shape
    N, L = mask.shape
    mask = mask.unsqueeze(1).repeat(1, R, 1)
    A[mask] = 0.0
    return A


_registered = []


def register(cls):
    _registered.append(cls)
    return cls


def torch_forward_diff(net):
    for cls in _registered:
        out = cls.create(net)
        if out is not None:
            return out
    assert False, type(net)


class FwdDiff(nn.Module):
    @classmethod
    def create(cls, net) -> "Optional[FwdDiff]":
        raise NotImplemented()

    def __init__(self, net):
        super().__init__()
        self.net = net

    def dual(self, x, dx) -> "Tuple[Tensor, Tensor]":
        # Must be batched.
        raise NotImplemented()

    def forward(self, x):
        N, L = x.shape
        dx = torch.eye(L, device=x.device, dtype=x.dtype)
        dx = dx.repeat(N, 1, 1)
        x, dx = self.dual(x, dx)
        return dx.squeeze(-1)


@register
class LinearFwdDiff(FwdDiff):
    @classmethod
    def create(cls, net):
        if isinstance(net, nn.Linear):
            return cls(net)
        return None

    def dual(self, x, dx):
        dx = dx @ self.net.weight.T
        x = self.net(x)
        return x, dx


@register
class ReLUFwdDiff(FwdDiff):
    @classmethod
    def create(cls, net):
        if isinstance(net, nn.ReLU):
            return cls(net)
        return None

    def dual(self, x, dx):
        dx = torch_col_zero(dx, x <= 0)
        x = self.net(x)
        return x, dx


@register
class SequentialFwdDiff(FwdDiff):
    @classmethod
    def create(cls, net):
        if isinstance(net, nn.Sequential):
            return cls(net)
        return None
    
    def __init__(self, net):
        super().__init__(net)
        self.fwds = nn.ModuleList(
            [torch_forward_diff(net_i) for net_i in net]
        )

    def dual(self, x, dx):
        for fwd in self.fwds:
            x, dx = fwd.dual(x, dx)
        return x, dx
