#!/usr/bin/env python3

import argparse
import os

import wandb
import yaml

from wandb_sweep_example import defs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("entity", type=str)
    args = parser.parse_args()

    entity = args.entity
    project = "test"

    sweep_config_file = os.path.join(defs.SOURCE_DIR, "simple_sweep.yaml")
    with open(sweep_config_file, "r") as f:
        sweep_config = yaml.safe_load(f)
    sweep_id = wandb.sweep(sweep=sweep_config, project=project)

    wandb.agent(sweep_id, entity=entity, project=project)

    print("[ Done ]")


if __name__ == "__main__":
    main()
