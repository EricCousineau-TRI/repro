import time

import numpy as np

import torch
from torch import nn


class TorchFakeLangevin(nn.Module):
    # Since we can't pass a net as argument, just wrap it into a module...
    def __init__(self, net):
        super().__init__()
        self.net = net

    def _gradient(self, yhs):
        out = self.net(yhs).sum()
        dy, = torch.autograd.grad([out], [yhs])
        assert dy is not None
        return dy.detach()

    def forward(self, yhs, num_iter: int):
        step_size = 0.001
        # WRONG (I think) - should detach in each loop.
        yhs = yhs.detach().requires_grad_(True)
        for i in range(num_iter):
            de_dact = self._gradient(yhs)
            yhs = yhs + de_dact * step_size
        return yhs


class TorchFakeLangevinPrint(nn.Module):
    # Same as above, but w/ print. Dunno how to make diff s.t. jit analysis will
    # use it.
    def __init__(self, net):
        super().__init__()
        self.net = net

    def _gradient(self, yhs):
        out = self.net(yhs).sum()
        dy, = torch.autograd.grad([out], [yhs])
        assert dy is not None
        return dy.detach()

    def forward(self, yhs, num_iter: int):
        step_size = 0.001
        yhs = yhs.detach().requires_grad_(True)
        for i in range(num_iter):
            print(i)
            de_dact = self._gradient(yhs)
            yhs = yhs + de_dact * step_size
        return yhs


@torch.no_grad()
def run(num_iter, use_print):
    print(f"num_iter={num_iter}, use_print={use_print}")

    N = 1
    L = 1
    DimY = 1
    hidden_sizes = []

    np.random.seed(0)
    yhs_init = np.random.rand(N * L, DimY).astype(np.float32)

    device = torch.device("cpu")
    net = nn.Linear(1, 1)
    net.eval().to(device)
    yhs = torch.from_numpy(yhs_init).to(device)

    if use_print:
        fast = TorchFakeLangevinPrint(net)
    else:
        fast = TorchFakeLangevin(net)

    fast = torch.jit.script(fast)

    def work():
        with torch.set_grad_enabled(True):
            return fast(yhs, num_iter).detach().cpu()

    try:
        # Warmup; needs >=2 for jit on first usage?
        for _ in range(2):
            yhs_new = work()

        t_start = time.time()
        # Simulate device transfer to flush graph.
        yhs_new = work().detach().cpu()
        dt = time.time() - t_start

        print(f"  Success: {dt:.3g}s")
    except RuntimeError as e:
        if "differentiated Tensors" in str(e):
            print(f"  Error: Unused diff'able param")
        else:
            raise


def main():
    run(7, False)  # <=7 does not throw.
    run(8, False)  # >=8 throws.
    run(8, True)   # Does not throw.


assert __name__ == "__main__"
main()
