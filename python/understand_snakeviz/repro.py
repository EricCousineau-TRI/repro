"""
Provides a dumb way to understand SnakeViz + pstats output.

I (Eric) dunno why horizontal times to sum to the total runtime.
"""

from contextlib import contextmanager
import cProfile as profile
import os
import pstats
import time


class Result:
    dt = None
    stats = None


@contextmanager
def do_profile():
    result = Result()

    pr = profile.Profile(builtins=False)
    pr.enable()

    t_start = time.time()
    yield result
    result.dt = time.time() - t_start

    pr.disable()
    result.stats = pstats.Stats(pr)


def sleep_L0():
    time.sleep(0.1)


def sleep_L1():
    time.sleep(0.1)
    sleep_L0()


def sleep_L2():
    time.sleep(0.1)
    sleep_L1()


def sleep_L0_2():
    time.sleep(0.5)


def sample_func():
    sleep_L0()
    sleep_L1()
    sleep_L2()
    sleep_L0_2()


def main():
    with do_profile() as result:
        sample_func()

    output_file = "/tmp/results.txt"

    result.stats.sort_stats("cumtime", "tottime").print_stats(30)
    result.stats.dump_stats(output_file)

    print("exec to snakeviz")
    os.execvp("snakeviz", ["snakeviz", "-s", output_file])


if __name__ == "__main__":
    main()
