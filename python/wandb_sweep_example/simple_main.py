#!/usr/bin/env python3

import argparse
import json
import sys

import wandb

from wandb_sweep_example import defs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_sweep_json", type=str, default=None)
    args = parser.parse_args()

    assert args.wandb_sweep_json is not None
    sweep_args = json.loads(args.wandb_sweep_json)
    assert sweep_args == {"custom_toggle": True}

    wandb.init()

    wandb.log({"val/loss": 1.0})

    if defs.should_fail():
        print("Failing on purpose")
        sys.exit(1)

    print("Success")


if __name__ == "__main__":
    main()
