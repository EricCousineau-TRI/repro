#!/usr/bin/env python3

# https://docs.wandb.ai/sweeps/quickstart

from contextlib import closing
import re
import subprocess
import time

import wandb

from wandb_pytorch_lightning_combo.process_util import CapturedProcessGroup

DT_INTERVAL = 0.05  # s


def run(procs):
    wandb_project = "uncategorized"

    sweep = procs.add(
        "sweep",
        [
            "wandb",
            "sweep",
            "--controller",
            "--project", wandb_project,
            "./train_wandb_pl_sweep_example.yaml",
        ],
    )

    # Wait for it to start up and print stuff out.
    sweep_id = None
    while sweep_id is None:
        stat = procs.poll()
        assert stat == {}, stat
        time.sleep(DT_INTERVAL)
        m = re.search(
            r"Created sweep with ID: (\w+)",
            sweep.output.get_text(),
        )
        if m is not None:
            sweep_id = m.group(1)

    print(f"Extracted sweep id: {sweep_id}")
    print(f"Run agent...")
    agent = procs.add(
        "agent", 
        [
            "wandb",
            "agent",
            "--project", wandb_project,
            sweep_id,
        ],
    )

    # Note: Agent hangs if the script encounters an error :/
    while agent.poll() is None:
        assert sweep.poll() is None
        time.sleep(DT_INTERVAL)
    assert agent.poll() == 0

    # Wait for sweep to close down.
    while sweep.poll() is None:
        time.sleep(DT_INTERVAL)

    stat = procs.poll()
    assert all(value == 0 for value in stat.values()), stat
    print("[ Done ]")


def main():
    procs = CapturedProcessGroup()
    with closing(procs):
        run(procs)


if __name__ == "__main__":
    main()
