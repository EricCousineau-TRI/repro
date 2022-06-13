#!/usr/bin/env python3

import argparse

import wandb
import yaml

from wandb_sweep_example import defs
from wandb_sweep_example.wandb_sweep_clean_and_resume import (
    main as resume_main,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("entity", type=str)
    args = parser.parse_args()

    entity = args.entity
    project = "test"

    with open(defs.SWEEP_CONFIG_FILE, "r") as f:
        sweep_config = yaml.safe_load(f)
    sweep_id = wandb.sweep(sweep=sweep_config, project=project)

    def run_agent():
        wandb.agent(sweep_id, entity=entity, project=project)

    print("Run failing agent")
    defs.set_should_fail(True)
    run_agent()

    print("Reset for success")
    defs.set_should_fail(False)

    print("restart sweep")
    resume_main(["-e", entity, "-p", project, "-y", sweep_id])

    # revive agent
    try:
        run_agent()
    except wandb.errors.CommError as e:
        # this will most likely happen?
        assert "Sweep" in str(e) and "is not running" in str(e)
        print("Please open sweep url and manually resume")
        print("Press ENTER when done")
        input()
        print("Rerunning")
        run_agent()

    print("[ Done ]")


if __name__ == "__main__":
    main()
