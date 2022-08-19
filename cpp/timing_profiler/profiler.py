from collections import defaultdict
import copy
import time

from timing_profiler import running_time_stats_map_t, running_time_stats_t
from timing_profiler.running_stats import RunningStats


def counter():
    return time.time()


class Timer:
    def __init__(self):
        self.reset()

    def reset(self):
        self._stats = RunningStats()
        self._start = None

    def start(self):
        self._start = counter()

    def stop(self):
        assert self._start is not None
        self.lap()
        self._start = None

    def lap(self):
        now = counter()
        if self._start is not None None:
            delta = now - self._start
            self._stats.add(delta)
        self._start = now

    def stats(self):
        return copy.deepcopy(self._stats)


class Profiler:
    def __init__(self):
        self._timers = defaultdict(Timer)

    def timer(self, name):
        return self._timers[name]

    def stats_map(self):
        return {
            name: timer.stats()
            for name, timer in self._timers.items()
        }

    def reset(self):
        for timer in self._timers.values():
            timer.reset()


def running_stats_to_lcm(timestamp, stats):
    assert isinstance(timestamp, int)
    assert isinstance(stats, RunningStats)
    message = running_time_stats_t()
    message.timestamp = timestamp
    message.count = stats.count()
    message.mean = stats.mean()
    message.m2 = stats.m2()
    message.min = stats.min()
    message.max = stats.max()
    if stats.mean() != 0.0:
        message.norm_stddev = stats.stddev() / stats.mean()
    else:
        message.norm_stddev = 0.0
    return message


def running_stats_map_to_lcm(timestamp, stats_map):
    assert isinstance(timestamp, int)
    assert isinstance(stats_map, dict)
    message = running_time_stats_map_t()
    message.count = len(stats_map)
    for name, stats in stats_map.items():
        message.names.append(name)
        message.stats.append(running_stats_to_lcm(timestamp, stats))
    return message


def publish_profiler_results(lcm, channel, profiler):
    assert isinstance(profiler, Profiler)
    stats_map = profiler.stats_map()
    timestamp = int(counter() * 1e6)
    message = running_stats_map_to_lcm(timestamp, stats_map)
    lcm.publish(channel, message.encode())
