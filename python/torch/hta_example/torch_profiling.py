from contextlib import contextmanager
import time
import json
from pathlib import Path
import shutil
import subprocess

import torch
from torch.profiler import (
    profile, ProfilerActivity, tensorboard_trace_handler,
    schedule,
)


def hta_patch_json(json_file):
    # Per warning from HTA
    # https://github.com/facebookresearch/HolisticTraceAnalysis/issues/107
    json_file = Path(json_file)
    with json_file.open("r") as f:
        obj = json.load(f)
    key = "distributedInfo"
    if key not in obj:
        obj[key] = {}
    if "rank" not in obj[key]:
        obj[key]["rank"] = 0
    with json_file.open("w") as f:
        json.dump(obj, f)


@contextmanager
def torch_profiling_basic(trace_dir, *, rm_before=True):
    # WARNING: trace_dir needs to be throw-away

    trace_dir = Path(trace_dir)
    tensorboard_handler = tensorboard_trace_handler(
        dir_name=trace_dir,
        # don't gzip b/c we need to patch :(
        use_gzip=False,
    )

    def trace_handler(prof):
        if rm_before and Path(trace_dir).is_dir():
            shutil.rmtree(trace_dir)
        trace_dir.mkdir(exist_ok=True)

        # Visualize flamegraphs
        # https://pytorch.org/docs/2.2/profiler.html#torch.profiler._KinetoProfile.export_stacks
        print("Manual export stack")
        cpu_trace = trace_dir / "stack_cpu.trace"
        cuda_trace = trace_dir / "stack_gpu.trace"
        prof.export_stacks(cpu_trace, metric="self_cpu_time_total")
        subprocess.run(
            f"flamegraph --title 'CPU time' --countname 'us.' {cpu_trace} > {cpu_trace}.svg",
            shell=True,
            check=True,
        )
        prof.export_stacks(cuda_trace, metric="self_cuda_time_total")
        subprocess.run(
            f"flamegraph --title 'GPU time' --countname 'us.' {cuda_trace} > {cuda_trace}.svg",
            shell=True,
            check=True,
        )
        print(f"  x-www-browser {cpu_trace}.svg {cuda_trace}.svg")

        print("Tensorboard handler")
        tensorboard_handler(prof)
        # Expect only 1 json.
        trace_json, = trace_dir.glob("*.json")
        hta_patch_json(trace_json)

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    with profile(
        activities=activities,
        on_trace_ready=trace_handler,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        # Ensure we actually get stacks.
        # https://github.com/pytorch/pytorch/issues/100253
        experimental_config=(
            torch._C._profiler._ExperimentalConfig(verbose=True)
        ),
    ) as prof:
        t_start = time.time()
        yield prof
        dt = time.time() - t_start
        print(f"Profile Elapsed: {dt:.5g} sec")
