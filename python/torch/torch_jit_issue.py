import time

import torch


class TorchFakeLangevin(torch.nn.Module):
    # Since we can't pass a net as argument, just wrap it into a module...
    def __init__(self, net):
        super().__init__()
        self.net = net
        self._step_size = 0.001

    def _on_iter(self, i: int):
        pass

    def _gradient(self, x):
        y = self.net(x)
        dy_dx, = torch.autograd.grad([y.sum()], [x])
        assert dy_dx is not None
        return dy_dx.detach()

    def forward(self, x, num_iter: int):
        # Bad?
        x = x.detach().requires_grad_(True)
        for i in range(num_iter):
            self._on_iter(i)
            dy_dx = self._gradient(x)
            x = x + dy_dx * self._step_size
        return x


class TorchFakeLangevinPrint(TorchFakeLangevin):
    # Same as above, but w/ print.
    # Dunno how to pass a func / bool without it affecting this behavior.
    def _on_iter(self, i: int):
        print(i)


class TorchFakeLangevinCorrect(TorchFakeLangevin):
    def forward(self, x, num_iter: int):
        for i in range(num_iter):
            self._on_iter(i)
            # Better?
            x_tmp = x.detach().requires_grad_(True)
            dy_dx = self._gradient(x_tmp)
            x = x + dy_dx * self._step_size
        return x


@torch.no_grad()
def run(num_iter, cls, use_jit=True):
    print(f"num_iter={num_iter}, cls={cls.__name__}, use_jit={use_jit}")
    x = torch.ones(1, 1)
    net = torch.nn.Linear(1, 1).eval()
    fast = cls(net)
    if use_jit:
        fast = torch.jit.script(fast)

    def work():
        with torch.set_grad_enabled(True):
            return fast(x, num_iter)

    try:
        work()
        print(f"  Success")
    except RuntimeError as e:
        if "differentiated Tensors" in str(e):
            print(f"  Error: Unused diff'able param")
        else:
            raise


def main():
    run(8, TorchFakeLangevinCorrect)  # Does not throw
    run(7, TorchFakeLangevin)  # <=7 does not throw.
    run(8, TorchFakeLangevin)  # >=8 throws, confusing why it's senstive.
    run(8, TorchFakeLangevin, use_jit=False)
    run(8, TorchFakeLangevinPrint)   # Does not throw.


assert __name__ == "__main__"
main()
