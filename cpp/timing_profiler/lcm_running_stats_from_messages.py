"""
Prints statistics published over LCM.
"""

import argparse
from collections import defaultdict, deque
import copy
from functools import partial
from threading import Thread
import time

from lcm import LCM

from timing_profiler import running_time_stats_map_t
from timing_profiler.common.lcm_util import lcm_handle_all
from timing_profiler.common.running_stats import RunningStats

IGNORE = {"PROFILER_RESET"}


def from_running_time_stats_t(message):
    stats = RunningStats()
    stats._count = message.count
    stats._mean = message.mean
    stats._m2 = message.m2
    stats._min = message.min
    stats._max = message.max
    return stats


def from_running_time_stats_map_t(message):
    out = dict()
    for name, sub in zip(message.names, message.stats):
        out[name] = from_running_time_stats_t(sub)
    return out


class Spy:
    def __init__(self, lcm, channels):
        self._timings = dict()
        self._subs = []
        for channel in channels:
            sub = lcm.subscribe(channel, self._callback)
            sub.set_queue_capacity(1)
            self._subs.append(sub)

    def _callback(self, channel, raw):
        if channel in IGNORE:
            return
        message = running_time_stats_map_t.decode(raw)
        self._timings[channel] = from_running_time_stats_map_t(message)

    def timings(self):
        """
        Returns Dict[str, Dict[RunningStats]] for each channel.
        """
        return copy.deepcopy(self._timings)


def print_timing_stats(timings):
    """Takes Dict[str, Timing] and pretty prints a simple table."""
    print("{:<30}{:>15}{:>15}{:>10}{:>15}{:>15}".format(
        "Label",
        "Mean Time (s)",
        "Stddev",
        "Samples",
        "Min (s)",
        "Max (s)",
    ))
    items = sorted(timings.items(), key=lambda x: x[0])
    for channel, timing in items:
        print(f"[ {channel} ]")
        sub_items = sorted(timing.items(), key=lambda x: x[0])
        for label, stats in sub_items:
            norm_stddev = stats.stddev()
            print("{:<30}{:>15.7g}{:>15.7g}{:>10}{:>15.7g}{:>15.7g}".format(
                label,
                stats.mean(),
                norm_stddev,
                stats.count(),
                stats.min(),
                stats.max(),
            ))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--lcm_url", type=str, default="",
        help="LCM URL to subscribe to.",
    )
    parser.add_argument(
        "--channels", type=str, nargs="*", default=[".*"],
        help="Stats channels to print out.",
    )
    parser.add_argument(
        "--dt_print_sec", type=float, default=0.5,
        help="How often to print stats tables to console.",
    )
    args = parser.parse_args()
    lcm = LCM(args.lcm_url)
    spy = Spy(lcm, args.channels)

    dt_print = args.dt_print_sec

    # Run loop.
    t_print = time.time()
    while True:
        lcm_handle_all(lcm, min_new_count=0)

        t = time.time()
        if t >= t_print:
            t_print += dt_print
            timings = spy.timings()
            print_timing_stats(timings)
            print("---")


if __name__ == "__main__":
    main()
