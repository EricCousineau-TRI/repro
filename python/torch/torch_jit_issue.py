import time

import torch


class TorchFakeLangevin(torch.nn.Module):
    # Since we can't pass a net as argument, just wrap it into a module...
    def __init__(self, net):
        super().__init__()
        self.net = net

    def _gradient(self, x):
        y = self.net(x).sum()
        dy_dx, = torch.autograd.grad([y], [x])
        assert dy_dx is not None
        return dy_dx.detach()

    def forward(self, x, num_iter: int):
        step_size = 0.001
        # WRONG (I think) - should detach in each loop.
        x = x.detach().requires_grad_(True)
        for i in range(num_iter):
            de_dact = self._gradient(x)
            x = x + de_dact * step_size
        return x


class TorchFakeLangevinPrint(torch.nn.Module):
    # Same as above, but w/ print. Dunno how to make diff s.t. jit analysis will
    # use it.
    def __init__(self, net):
        super().__init__()
        self.net = net

    def _gradient(self, x):
        y = self.net(x).sum()
        dy_dx, = torch.autograd.grad([y], [x])
        assert dy_dx is not None
        return dy_dx.detach()

    def forward(self, x, num_iter: int):
        step_size = 0.001
        x = x.detach().requires_grad_(True)
        for i in range(num_iter):
            print(i)
            de_dact = self._gradient(x)
            x = x + de_dact * step_size
        return x


@torch.no_grad()
def run(num_iter, use_print):
    print(f"num_iter={num_iter}, use_print={use_print}")

    x = torch.ones(1, 1)
    net = torch.nn.Linear(1, 1).eval()

    if use_print:
        fast = TorchFakeLangevinPrint(net)
    else:
        fast = TorchFakeLangevin(net)

    fast = torch.jit.script(fast)

    def work():
        with torch.set_grad_enabled(True):
            return fast(x, num_iter).detach().cpu()

    try:
        # Warmup; needs >=2 for jit on first usage?
        for _ in range(2):
            work()

        t_start = time.time()
        work()
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
