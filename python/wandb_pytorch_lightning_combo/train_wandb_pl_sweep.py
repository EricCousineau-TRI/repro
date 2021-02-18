#!/usr/bin/env python3

# https://docs.wandb.ai/sweeps/quickstart

import re
import subprocess

import wandb


def main():
    result = subprocess.run(
        ["wandb", "sweep", "./wandb_sweep.yaml"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        check=True,
    )
    print(result.stdout)
    sweep_id = re.search(r"^sweep id: (\w+)$", result.stdout, flags=re.MULTILINE)
    print()
    print(f"sweep id: {sweep_id}")

    print("Run agent")
    subprocess.run(
        ["wandb", "agent", sweep_id],
        check=True,
    )


if __name__ == "__main__":
    main()
