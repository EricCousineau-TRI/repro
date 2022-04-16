#!/usr/bin/env python3

import argparse
import json

import wandb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_sweep_json", type=str, required=True)
    args = parser.parse_args()

    sweep_args = json.loads(args.wandb_sweep_json)
    assert len(sweep_args) == 1
    buggy = sweep_args["buggy"]

    if buggy:
        resume = "must"
    else:
        resume = None

    wandb.init(resume=resume)

    wandb.log({"fake_loss": 1.0})
    print("Done")


if __name__ == "__main__":
    main()
