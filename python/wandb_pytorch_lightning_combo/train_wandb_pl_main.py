#!/usr/bin/env python3

import argparse

import numpy as np
import wandb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_wandb_sweep", action="store_true")
    parser.add_argument("--wandb_sweep_json", type=str, default=None)
    args = parser.parse_args()

    if args.is_wandb_sweep:
        print(f"I am a sweep! {args.wandb_sweep_json}")
        assert args.wandb_sweep_json is not None
        assert args.wandb_sweep_json != ""

    wandb.init(
        name="test-run",
        project="uncategorized",
    )
    loss = np.array(1.0)
    wandb.log({"val/loss": loss})
    print("[ Done ]")


if __name__ == "__main__":
    main()
