"""
Stuff adapted from Anzu code...
"""

import cProfile as profile
import pstats
import random
import time

import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import numpy as np
import torch
import torchvision


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


class Setup:
    N: int = 1
    C: int = 3
    D: int = 12
    W: int = 640
    H: int = 480

    @torch.no_grad()
    def sample(self, device):
        rgb_tensor = torch.rand(
            (self.N, self.C, self.H, self.W), device=device,
        )
        return rgb_tensor


def make_fcn_resnet50(num_classes):
    model = torchvision.models.segmentation.fcn_resnet50(
        pretrained=False,
        pretrained_backbone=False,
        progress=False,
        num_classes=num_classes,
        aux_loss=None,
    )
    return model


@torch.no_grad()
def main():
    seed(0)

    gpu = torch.device("cuda")
    cpu = torch.device("cpu")

    setup = Setup()

    net = make_fcn_resnet50(num_classes=setup.D)
    net.eval().to(gpu)

    count = 10
    stats = None
    dts = []
    for i in range(count):
        seed(0)
        rgb_tensor = setup.sample(device=cpu)
  
        is_last = i + 1== count
        if is_last:
            pr = profile.Profile()
            pr.enable()
        t_start = time.time()

        # Simulate transfer.
        rgb_tensor = rgb_tensor.to(gpu)
        y = net(rgb_tensor)
        dd_tensor = y["out"]
        dd_array = dd_tensor.to(cpu).numpy()
        assert dd_array is not None

        dt = time.time() - t_start
        dts.append(dt)
        if is_last:
            pr.disable()
            stats = pstats.Stats(pr)

    stats.sort_stats("cumtime").strip_dirs().print_stats(5)

    print()
    dt_mean = np.mean(dts)
    print(f"dt_mean: {dt_mean:.4f}s")
    print(f"dts[-1]: {dts[-1]:.4f}s")


if __name__ == "__main__":
    main()
