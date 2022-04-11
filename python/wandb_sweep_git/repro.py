#!/usr/bin/env python3

import argparse

import wandb
import yaml

from wandb_sweep_git import defs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("entity", type=str)
    args = parser.parse_args()

    entity = args.entity
    project = "test"

    with open(defs.SWEEP_CONFIG_FILE, "r") as f:
        sweep_config = yaml.safe_load(f)
    sweep_id = wandb.sweep(sweep=sweep_config, project=project)

    print("Run agent")
    wandb.agent(sweep_id, entity=entity, project=project)

    print("[ Done ]")


if __name__ == "__main__":
    main()
