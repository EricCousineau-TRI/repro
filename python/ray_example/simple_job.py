#!/usr/bin/env python3

import argparse
import time

import ray


@ray.remote(num_cpus=2)
def simple_job():
    time.sleep(5)
    return 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local", action="store_true", help="If running in a local machine"
    )
    args = parser.parse_args()

    if args.local:
        ray.init(address="")
    else:
        ray.init(address="auto")

    ray.get(
        [
            simple_job.remote()
            for _ in range(10)
        ]
    )


if __name__ == "__main__":
    main()
