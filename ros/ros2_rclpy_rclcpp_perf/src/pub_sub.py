import ctypes
import functools
import multiprocessing as mp
import os
import signal
import sys
import time

import rclpy
from geometry_msgs.msg import PoseStamped as Message

from running_stats import TimingStats, header_timing_stats, format_timing_stats

RATE_HZ = 2000.0
# https://stackoverflow.com/a/60153370/7829525
DEFAULT_TIMERSLACK = 50e-6


def wrap_rclpy(func, *args, **kwargs):
    rclpy.init(signal_handler_options=rclpy.SignalHandlerOptions.NO)
    try:
        func(*args, **kwargs)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


def pub_main(count=1, ready_flag=None):
    node = rclpy.create_node("pub")
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)

    pub = [None] * count
    stats = [None] * count
    for i in range(count):
        pub[i] = node.create_publisher(Message, f"/message_{i}", 1)
        stats[i] = TimingStats()

    rate = LoopRate(RATE_HZ)
    print(f"Pub running, target rate: {1 / RATE_HZ}")
    t_total_start = time.perf_counter()
    if ready_flag is not None:
        ready_flag.value = 1
        # Wait for ack.
        while ready_flag.value == 1:
            time.sleep(0.001)
    try:
        while rclpy.ok():
            for i in range(count):
                pub[i].publish(Message())
                stats[i].tick()
            # # N.B. Uncommenting this impacts performance, even if just for 50us?
            # executor.spin_once(timeout_sec=DEFAULT_TIMERSLACK)
            rate.sleep()
    finally:
        dt_total = time.perf_counter() - t_total_start
        lines = [
            f"Pub done after {dt_total:.5f} sec",
            format_stats("pub.", stats),
            "",
        ]
        print("\n".join(lines), file=sys.stderr)
        node.destroy_node()


def sub_main(count=1, ready_flag=None):
    node = rclpy.create_node("sub")
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)

    stats = [None] * count
    sub = [None] * count

    def callback(i, msg):
        stats[i].tick()

    for i in range(count):
        stats[i] = TimingStats()
        callback_i = functools.partial(callback, i)
        sub[i] = node.create_subscription(
            Message, f"/message_{i}", callback_i, 1
        )

    print(f"Sub running")
    t_total_start = time.perf_counter()
    if ready_flag is not None:
        ready_flag.value = 1
        # Wait for ack.
        while ready_flag.value == 1:
            time.sleep(0.001)
    try:
        while rclpy.ok():
            executor.spin_once(timeout_sec=DEFAULT_TIMERSLACK)
    finally:
        dt_total = time.perf_counter() - t_total_start
        lines = [
            f"Sub done after {dt_total:.5f} sec",
            format_stats("sub.", stats),
            "",
        ]
        print("\n".join(lines), file=sys.stderr)
        node.destroy_node()


class LoopRate:
    def __init__(self, hz):
        self.dt = 1.0 / hz
        self.reset()

    def reset(self):
        self.t_start = time.perf_counter()
        self.t_next = self.t_start + self.dt

    def sleep(self, *, dt_sleep=DEFAULT_TIMERSLACK):
        while time.perf_counter() < self.t_next:
            time.sleep(dt_sleep)
        # Choose next dt.
        self.t_next += self.dt
        if self.t_next < time.perf_counter():
            # Reset if we miss any ticks.
            self.t_next = time.perf_counter() + self.dt


def format_stats(prefix, stats):
    count = len(stats)
    fmt_stats = "{:<15}{}"
    header_text = header_timing_stats()
    lines = []
    lines.append(fmt_stats.format("", header_text))
    for i in range(count):
        text = format_timing_stats(stats[i].stats)
        lines.append(fmt_stats.format(f"{prefix}message_{i}", text))
    return "\n".join(lines)


class MpProcessGroup:
    def __init__(self, procs):
        self.procs = procs

    def __iter__(self):
        return iter(self.procs)

    def start(self):
        for proc in self.procs:
            proc.start()

    def poll(self):
        for proc in self.procs:
            assert proc.is_alive()

    def close(self):
        for proc in self.procs:
            if proc.is_alive():
                os.kill(proc.pid, signal.SIGINT)
        for proc in self.procs:
            # pass
            # Can hang here?
            proc.join(timeout=0.01)


def main():
    count = 7
    duration_sec = 1.0
    dt_rate = 1 / RATE_HZ

    pub_ready = mp.Value(ctypes.c_int)
    pub_ready.value = 0
    sub_ready = mp.Value(ctypes.c_int)
    sub_ready.value = 0

    procs = MpProcessGroup([
        mp.Process(target=wrap_rclpy, args=[pub_main, count, pub_ready]),
        mp.Process(target=wrap_rclpy, args=[sub_main, count, sub_ready]),
    ])
    for proc in procs:
        proc.daemon = True
    procs.start()

    while True:
        if pub_ready.value != 0 and sub_ready.value != 0:
            break
        time.sleep(dt_rate)
    pub_ready.value += 1
    sub_ready.value += 1

    print(f"Pub + sub ready. Stopping after {duration_sec}")
    print()

    t_end = time.time() + duration_sec
    try:
        while time.time() < t_end:
            procs.poll()
            time.sleep(dt_rate)
    except KeyboardInterrupt:
        pass
    finally:
        procs.close()


if __name__ == "__main__":
    main()
