"""
This is a more "easy-to-digest" form of the statistics from `lcm-spy`. Most
notably, "jitter" (whose meaning isn't clear) is now replaced with computed
variance.
"""

import argparse
from collections import defaultdict, deque
import copy
from functools import partial
import gc
from io import StringIO
import sys
from threading import Thread
import time

from lcm import LCM

from timing_profiler.common.lcm_util import lcm_handle_all
from timing_profiler.common.running_stats import RunningStats

IGNORE = {"LCM_SELF_TEST", "PROFILER_RESET"}


class Timing:
    """Records timing for a *concrete* channel."""
    def __init__(self):
        self.reset()

    def reset(self):
        self._last_time = None
        self._stats = RunningStats()

    def tick(self):
        """Update timing statistics."""
        now = time.time()
        if self._last_time is not None:
            dt = now - self._last_time
            self._stats.add(dt)
        self._last_time = now

    def stats(self):
        """Retuns reference to statistics."""
        return self._stats


class Spy:
    """
    Given an LCM instance a set of channel patterns, will compute
    running statistics for the duty cycle for receive times of the given
    messages.
    """
    def __init__(self, lcm, channels, *, enable_reset=True):
        if enable_reset:
            lcm.subscribe("PROFILER_RESET", self._reset)
        self._timings = defaultdict(Timing)
        self._subs = []
        for channel in channels:
            assert channel not in IGNORE
            sub = lcm.subscribe(channel, self._callback)
            sub.set_queue_capacity(1000)
            self._subs.append(sub)

    def _callback(self, channel, _):
        if channel in IGNORE:
            return
        self._timings[channel].tick()

    def _reset(self, channel, _):
        for timing in self._timings.values():
            timing.reset()

    def copy_timings(self):
        """
        Returns Dict[str, Timing] for each channel. This is copied for use in
        threading.
        """
        return copy.deepcopy(self._timings)


def print_timing_stats(timings, *, file=sys.stdout):
    """Takes Dict[str, Timing] and pretty prints a simple table."""
    print("{:<20}{:>15}{:>15}{:>10}{:>15}{:>15}".format(
        "Channel",
        "Mean Time (s)",
        "Stddev",
        "Samples",
        "Min (s)",
        "Max (s)",
    ), file=file)
    items = sorted(timings.items(), key=lambda x: x[0])
    for channel, timing in items:
        stats = timing.stats()
        print("{:<20}{:>15.7g}{:>15.7g}{:>10}{:>15.7g}{:>15.7g}".format(
            channel,
            stats.mean(),
            stats.stddev(),
            stats.count(),
            stats.min(),
            stats.max(),
        ), file=file)


def print_loop(timings_queue, dt_print):
    """Loop for printing timing tables."""
    t_print = time.time()
    dt_sleep = 0.001
    while True:
        t = time.time()
        if t >= t_print and len(timings_queue):
            f = StringIO()
            t_print += dt_print
            timings = timings_queue.popleft()
            print("---", file=f)
            print_timing_stats(timings, file=f)
            print(f.getvalue(), flush=True, end="")
        time.sleep(dt_sleep)


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
        help="List of channel regex patterns to subscribe to.",
    )
    parser.add_argument(
        "--dt_print_sec", type=float, default=0.5,
        help="How often to print timing table to console.",
    )
    args = parser.parse_args()
    lcm = LCM(args.lcm_url)
    spy = Spy(lcm, args.channels)
    # Delegate printing to the non-main thread, so it's easy to allow for
    # command-line level prioritizaion (`taskset`, `chrt`) to be admitted for
    # processing.
    timings_queue = deque([], maxlen=1)
    print_target = partial(
        print_loop, timings_queue, dt_print=args.dt_print_sec
    )
    # n.b. sometimes this still dies in weird race condition ways.
    print_thread = Thread(target=print_target, daemon=True)
    print_thread.start()

    gc.disable()
    # Run loop.
    while True:
        lcm_handle_all(lcm, min_new_count=0)
        timings_queue.append(spy.copy_timings())
        time.sleep(50e-6)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
