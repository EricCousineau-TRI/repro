#!/usr/bin/env python
import pickle

import numpy as np


def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def save_pickle(filename, obj):
    with open(filename, "wb") as f:
        return pickle.dump(obj, f)


def take_first(xs):
    return next(iter(xs))


def main():
    workspace = load_pickle("./workspace.pkl")
    task = take_first(workspace["tasks"].values())
    task_min = {
        "task_cost_A": task["task_cost_A"],
        "task_cost_b": task["task_cost_b"],
        "task_cost_proj": task["task_cost_proj"],
    }
    workspace_min = {
        "M": workspace["M"],
        "C": workspace["C"],
        "tau_g": workspace["tau_g"],
        "task": task_min,
    }
    save_pickle("./workspace_min.pkl", workspace_min)


if __name__ == "__main__":
    main()
