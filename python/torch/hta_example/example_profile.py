import os
from pathlib import Path
import time

import torch

from python_profiling import use_py_spy
from torch_profiling import torch_profiling_basic


@torch.no_grad()
def make_model_and_data():
    model = torch.hub.load(
        'pytorch/vision:v0.10.0', 'resnet18', pretrained=False
    )
    W, H = 1024, 768
    N, C = 10, 3
    data = torch.zeros((N, C, H, W), dtype=torch.float32)
    return model, data


def main():
    # This makes simpler profilers like py-spy work better. But, ofc, this
    # causes things to be less efficient (e.g., HTA reports more idle time).
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    model, data_cpu = make_model_and_data()
    device = torch.device("cuda")
    model.to(device)

    @torch.no_grad()
    def do_work():
        data = data_cpu.to(device)
        output = model(data)
        torch.cuda.synchronize()
        output_cpu = output.cpu()

    do_work()  # warmup
    do_work()

    t_start = time.time()
    do_work()
    dt = time.time() - t_start
    print(f"Nominal Elapsed: {dt:.5g} sec")

    count = 10

    with use_py_spy():
        for _ in range(count):
            do_work()

    trace_dir = Path("~/tmp/tensorboard_trace").expanduser()
    with torch_profiling_basic(trace_dir) as prof:
        for _ in range(count):
            do_work()
            prof.step()


if __name__ == "__main__":
    main()
