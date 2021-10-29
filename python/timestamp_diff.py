#!/usr/bin/env python3

import argparse
import datetime
import os

import numpy as np


def parse_time(file):
    # Assumes ends with ISO-8601.
    prefix = "2021-"
    file = file.rstrip("/")
    base = os.path.basename(file)
    s = base[base.index(prefix):]
    s = s[:-len("-04-00")]
    fmt = "%Y-%m-%dT%H-%M-%S"
    return datetime.datetime.strptime(s, fmt).timestamp()


def print_delta(ts):
    dts = np.diff(ts)
    dt_span = ts[-1] - ts[0]
    print(
        f"count: {len(ts)}\n"
        f"dt_span: {dt_span}\n"
        f"dt_mean: {np.mean(dts)}\n"
        f"dt_min: {np.min(dts)}\n"
        f"dt_max: {np.max(dts)}\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=str)
    parser.add_argument(
        "--mode", type=str, choices=["mtime", "parse"], default="mtime",
    )
    args = parser.parse_args()

    if args.mode == "mtime":
        extract_t = os.path.getmtime
    elif args.mode == "parse":
        extract_t = parse_time
    else:
        assert False
    ts = [extract_t(file) for file in args.files]
    ts.sort()
    print_delta(ts)


if __name__ == "__main__":
    main()
