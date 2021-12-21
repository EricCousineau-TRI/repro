from contextlib import contextmanager

import torch


@contextmanager
def disable_param_grad(net):
    restore_grad = [param.requires_grad for param in net.parameters()]
    yield
    for param, restore_grad_i in zip(net.parameters(), restore_grad):
        param.requires_grad = restore_grad_i
