#!/usr/bin/env python3

import argparse
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


def pub_main(rate_hz, count, ready_flag):
    node = rclpy.create_node("pub")
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)

    pub = [None] * count
    stats = [None] * count
    for i in range(count):
        pub[i] = node.create_publisher(Message, f"/message_{i}", 1)
        stats[i] = TimingStats()

    rate = LoopRate(rate_hz)
    print(f"Pub running, target rate: {rate_hz}")
    t_total_start = time.perf_counter()

    cur = ready_flag.value + 1
    ready_flag.value = cur
    # Wait for ack.
    while ready_flag.value == cur:
        time.sleep(DEFAULT_TIMERSLACK)

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
        # Wait for signal to print and finish.
        cur = ready_flag.value
        while ready_flag.value == cur:
            time.sleep(DEFAULT_TIMERSLACK)
        lines = [
            f"Pub done after {dt_total:.5f} sec",
            format_stats("pub.", stats),
            "",
        ]
        print("\n".join(lines), file=sys.stderr)
        ready_flag.value += 1
        node.destroy_node()


def sub_main(count, ready_flag):
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

    cur = ready_flag.value + 1
    ready_flag.value = cur
    # Wait for ack.
    while ready_flag.value == cur:
        time.sleep(DEFAULT_TIMERSLACK)

    try:
        while rclpy.ok():
            executor.spin_once(timeout_sec=DEFAULT_TIMERSLACK)
    finally:
        dt_total = time.perf_counter() - t_total_start
        # Wait for signal to print and finish.
        cur = ready_flag.value
        while ready_flag.value == cur:
            time.sleep(DEFAULT_TIMERSLACK)
        lines = [
            f"Sub done after {dt_total:.5f} sec",
            format_stats("sub.", stats),
            "",
        ]
        print("\n".join(lines), file=sys.stderr)
        ready_flag.value += 1
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
    def __init__(self, procs, *, daemon=True):
        self.procs = procs
        for proc in self.procs:
            proc.daemon = daemon

    def __iter__(self):
        return iter(self.procs)

    def start(self):
        for proc in self.procs:
            proc.start()

    def poll(self):
        for proc in self.procs:
            assert proc.is_alive()

    def close(self, *, sig=signal.SIGINT):
        for proc in self.procs:
            if proc.is_alive():
                os.kill(proc.pid, signal.SIGINT)
        for proc in self.procs:
            # Er, not sure if this does anything.
            proc.join(timeout=DEFAULT_TIMERSLACK)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--count", type=int, default=5, help="Num pubs / subs"
    )
    parser.add_argument(
        "--duration_sec",
        type=float,
        default=1.0,
        help="Seconds to run pub + sub",
    )
    parser.add_argument(
        "--rate_hz",
        type=float,
        default=2000.0,
        help="Rate (hz)",
    )
    args = parser.parse_args()

    count = args.count
    duration_sec = args.duration_sec
    rate_hz = args.rate_hz

    pub_ready = mp.Value(ctypes.c_int)
    pub_ready.value = 0
    sub_ready = mp.Value(ctypes.c_int)
    sub_ready.value = 0

    # Use multi-processing so that we can ensure ROS pubs don't try to
    # use shared mem (I think).
    procs = MpProcessGroup([
        mp.Process(
            target=wrap_rclpy, args=[pub_main, rate_hz, count, pub_ready]
        ),
        mp.Process(
            target=wrap_rclpy, args=[sub_main, count, sub_ready]
        ),
    ])
    procs.start()

    rate = LoopRate(rate_hz)
    # Ensure we wait until both are ready to start.
    while pub_ready.value == 0 or sub_ready.value == 0:
        rate.sleep()
    # Ack we're ready.
    pub_ready.value += 1
    sub_ready.value += 1

    print(f"Pub + sub ready. Stopping after {duration_sec}")
    print()

    t_end = time.time() + duration_sec
    rate.reset()
    try:
        while time.time() < t_end:
            procs.poll()
            rate.sleep()
    except KeyboardInterrupt:
        pass
    finally:
        procs.close()

        cur = pub_ready.value + 1
        pub_ready.value = cur
        rate.reset()
        while pub_ready.value == cur:
            rate.sleep()

        cur = sub_ready.value + 1
        sub_ready.value = cur
        rate.reset()
        while sub_ready.value == cur:
            rate.sleep()

        print()


if __name__ == "__main__":
    main()
