"""
Torch utilities to fit friction model coefficients to data.

The coordinate change approach is adapted from:
  Sutanto, Giovanni, Austin S Wang, Yixin Lin, Mustafa Mukadam, Gaurav S
  Sukhatme, Akshara Rai, and Franziska Meier.
  "Encoding Physical Constraints in Differentiable Newton-Euler Algorithm"
  https://arxiv.org/abs/2001.08861
  https://github.com/facebookresearch/differentiable-robot-model
"""

from functools import partial
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import tqdm

from anzu.friction import (
    arctanh,
    calc_joint_dry_friction,
    regularizer_tanh,
)


class UnconstrainedScalar(nn.Module):
    def __init__(self, x):
        super().__init__()
        x = torch.as_tensor(x)
        self.x = nn.Parameter(x)

    @property
    def data(self):
        return self.x.data

    @data.setter
    def data(self, data):
        self.x.data = data

    def forward(self):
        return self.x


class PositiveScalar(nn.Module):
    def __init__(self, x, b=0):
        super().__init__()
        self.b = b
        x = torch.as_tensor(x)
        s = torch.sqrt(x - b)
        self.s = nn.Parameter(s)
        torch.testing.assert_close(self(), x)

    def forward(self):
        return self.s**2 + self.b


def interpish(s):
    return (torch.tanh(s) + 1) / 2.0


def interpish_inv(x):
    y = x * 2.0 - 1
    return arctanh(y, np=torch)


class RangeConstrainedScalar(nn.Module):
    def __init__(self, x=0, *, a, b, c=1.0):
        super().__init__()
        self.a = a
        self.b = b
        self.c = c
        x = torch.as_tensor(x)
        cs = (x - a) / (b - a)
        # What about offsetting instead of solving?
        s = interpish_inv(cs) / c
        self.s = nn.Parameter(s)
        torch.testing.assert_close(self(), x)

    def forward(self):
        cs = self.c * self.s
        x = self.a + (self.b - self.a) * interpish(cs)
        return x


class JointDryFriction(nn.Module):
    @staticmethod
    def make_initial_guess(N=()):
        # N.B. Need non-zero value for u-max.
        ones = torch.ones(N)
        return JointDryFriction(v0=ones, u_max=1e-8 * ones)

    def __init__(self, *, v0, u_max, m=None, v0_min=1e-8):
        super().__init__()
        # Choose coordinates that are friendly to unconstrained optimization
        # a la `differentiable-robot-model`.
        self.u_max = PositiveScalar(u_max)
        self.v0 = PositiveScalar(v0, b=v0_min)
        if m is None:
            self.m = None
        else:
            self.m = RangeConstrainedScalar(m, a=0.4, b=0.99, c=10.0)

    def remove_projection(self):
        """Removes parameter space changes."""
        self.u_max = UnconstrainedScalar(self.u_max().detach())
        self.v0 = UnconstrainedScalar(self.v0().detach())
        if self.m is not None:
            self.m = UnconstrainedScalar(self.m().detach())

    def forward(self, v):
        u_max = self.u_max()
        v0 = self.v0()
        if self.m is None:
            m = None
        else:
            m = self.m()
        regularizer = partial(
            regularizer_tanh,
            m=m,
            np=torch,
        )
        return calc_joint_dry_friction(v, v0, regularizer, u_max)

    def custom_named_parameters(self):
        # Allows us to get "bundled" parameters, not just the lowest level
        # names. Note that these should be used with `param_value()`.
        out = dict(u_max=self.u_max, v0=self.v0)
        if self.m is not None:
            out.update(m=self.m)
        return out


class JointFriction(nn.Module):
    @staticmethod
    def make_initial_guess(N=()):
        # N.B. Need non-zero value for damping.
        ones = torch.ones(N)
        return JointFriction(
            dry=JointDryFriction.make_initial_guess(N),
            b=ones * 1e-8,
        )

    def __init__(self, *, dry, b):
        super().__init__()
        self.dry = dry
        self.b = PositiveScalar(b)

    def forward(self, v):
        b = self.b()
        tau_dry = self.dry(v)
        tau_viscous = -b * v
        return tau_dry + tau_viscous

    def custom_named_parameters(self):
        dry = self.dry.custom_named_parameters()
        out = {f"dry.{k}": v for k, v in dry.items()}
        out.update(b=self.b)
        return out


def param_value(p):
    """
    Extract value from a parameter / argument-less module (projected
    parameter).
    """
    if isinstance(p, nn.Module):
        return p()
    else:
        return p.data


@torch.no_grad()
def run_fit_experiment(*, vs, us, friction, ensure_valid_param, lr):
    """
    Attempts to fit a dry + viscous friction estimate to (noisy) simulation
    data.

    Given the following equations of motion:
        M * vd + C(q, v) + g(q) = τ_id + τ_friction + τ_other
    where `τ_id` captures "presently known" inverse dynamics, this attempts to
    regress against measured / estimated τ_friction.

    Warning:
         It is important that `τ_id` term does *not* capture joint-level
         friction (viscous and/or dry).

    Note:
        You can use an estimated "external torques" if you have a *great*
        inertial model for `τ_id` and/or you trust a robot driver's estimate.
        If you are not sure, then you may want to physically attempt to remove
        any confounding signals (e.g. orient a joint against gravity, check
        that the center of mass is not too eccentric, etc.).

    Arguments:
        vs: Velocities
        us: Friction torques representing `τ_friction` above.
        friction: JointDryFriction to optimize.
        ensure_valid_param:
            Use to check / mutate parameters to be valid after optimization
            step.
        lr: Learning rate for Adam.
    """

    num_epoch = 300
    num_epoch_per_plot = num_epoch / 5

    opt = torch.optim.Adam(friction.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    dataset = torch.stack([vs, us])
    train_loader = DataLoader(dataset, batch_size=128, shuffle=False)

    plot_info = []
    loss_info = []

    torch.set_grad_enabled(True)

    @torch.no_grad()
    def add_plot_info(i):
        us_pred_i = friction(vs)
        plot_info.append((i, us_pred_i))

    # Add first plot before first training steps.
    add_plot_info(0)

    epoch_iter = tqdm.tqdm(range(num_epoch))
    for i in epoch_iter:
        losses_i = []
        for batch in train_loader:
            v, u_gt = batch
            # Predict.
            u = friction(v)
            # Optimize.
            loss = loss_fn(u_gt, u)
            loss.backward()
            opt.step()
            opt.zero_grad()
            # Check / correct stuff.
            ensure_valid_param(friction)
            # Record.
            losses_i.append(loss.item())
        loss_epoch = np.mean(losses_i)
        loss_info.append((i, loss_epoch))
        epoch_iter.set_postfix({"loss": loss_epoch})
        # Record some plotting data.
        if (i + 1) % num_epoch_per_plot == 0:
            add_plot_info(i)

    return SimpleNamespace(
        vs=vs,
        us=us,
        plot_info=plot_info,
        loss_info=loss_info,
    )


@torch.no_grad()
def plot_fit_experiment(info):
    plt.figure(num=1)
    plt.plot(info.vs.numpy(), info.us.numpy(), label="data")
    for i, us_pred in info.plot_info:
        plt.plot(info.vs.numpy(), us_pred.numpy(), label=f"epoch {i}")
    plt.legend()

    plt.figure(num=2)
    epochs, loss_per_epoch = np.asarray(info.loss_info).T
    plt.plot(epochs, loss_per_epoch)
