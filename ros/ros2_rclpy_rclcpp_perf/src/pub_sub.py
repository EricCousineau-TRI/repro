import ctypes
import functools
import multiprocessing as mp
import os
import signal
import time

import rclpy
from ros2_rclpy_rclcpp_perf.msg import ExampleStatus, ExampleCommand

from running_stats import TimingStats, header_timing_stats, format_timing_stats

RATE_HZ = 1000.0


def make_status():
    msg = ExampleStatus()
    return msg


def make_command():
    msg = ExampleCommand()
    return msg


def wrap_rclpy(func, *args, **kwargs):
    rclpy.init(signal_handler_options=rclpy.SignalHandlerOptions.NO)
    try:
        func(*args, **kwargs)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


def pub_main(count=1, ready_flag=None):
    node = rclpy.create_node("pub")
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)

    pub_status = [None] * count
    stats_status = [None] * count
    pub_command = [None] * count
    stats_command = [None] * count
    for i in range(count):
        pub_status[i] = node.create_publisher(
            ExampleStatus, f"/status_{i}", 1
        )
        stats_status[i] = TimingStats()
        pub_command[i] = node.create_publisher(
            ExampleCommand, f"/command_{i}", 1
        )
        stats_command[i] = TimingStats()

    rate = LoopRate(RATE_HZ)
    print("Pub running")
    t_total_start = time.perf_counter()
    if ready_flag is not None:
        ready_flag.value = 1
    try:
        while rclpy.ok():
            for i in range(count):
                pub_status[i].publish(make_status())
                stats_status[i].tick()
                pub_command[i].publish(make_command())
                stats_command[i].tick()
            executor.spin_once(timeout_sec=50e-6)
            rate.sleep()
    finally:
        dt_total = time.perf_counter() - t_total_start
        print(f"Pub done after {dt_total:.3g} sec")
        print_stats_status_and_command("pub.", stats_status, stats_command)
        print()


def sub_main(count=1, ready_flag=None):
    node = rclpy.create_node("sub")
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)

    stats_status = [None] * count
    stats_command = [None] * count

    def callback_status(i, msg):
        stats_status[i].tick()

    def callback_command(i, msg):
        stats_command[i].tick()

    sub_status = [None] * count
    sub_command = [None] * count
    for i in range(count):
        callback_status_i = functools.partial(callback_status, i)
        stats_status[i] = TimingStats()
        sub_status[i] = node.create_subscription(
            ExampleStatus, f"/status_{i}", callback_status_i, 1
        )
        callback_command_i = functools.partial(callback_command, i)
        stats_command[i] = TimingStats()
        sub_command[i] = node.create_subscription(
            ExampleCommand, f"/command_{i}", callback_command_i, 1
        )

    rate = LoopRate(RATE_HZ)
    print("Sub running")
    t_total_start = time.perf_counter()
    if ready_flag is not None:
        ready_flag.value = 1
    try:
        while rclpy.ok():
            executor.spin_once(timeout_sec=50e-6)
            rate.sleep()
    finally:
        dt_total = time.perf_counter() - t_total_start
        print(f"Sub done after {dt_total:.3g} sec")
        print_stats_status_and_command("sub.", stats_status, stats_command)
        print()


class LoopRate:
    def __init__(self, hz):
        self.dt = 1.0 / hz
        self.reset()

    def reset(self):
        self.t_start = time.perf_counter()
        self.t_next = self.t_start + self.dt

    def sleep(self, *, dt_sleep=1e-4):
        while time.perf_counter() < self.t_next:
            time.sleep(dt_sleep)
        # Choose next dt.
        self.t_next += self.dt
        if self.t_next < time.perf_counter():
            # Reset if we miss any ticks.
            self.t_next = time.perf_counter() + self.dt


def print_stats_status_and_command(prefix, stats_status, stats_command):
    count = len(stats_status)
    assert len(stats_command) == count
    fmt_stats = "{:>10}{}"
    header_text = header_timing_stats()
    lines = []
    lines.append(fmt_stats.format("", header_text))
    for i in range(count):
        status_text = format_timing_stats(stats_status[i].stats)
        lines.append(fmt_stats.format(f"{prefix}status[{i}]", status_text))
        command_text = format_timing_stats(stats_command[i].stats)
        lines.append(fmt_stats.format(f"{prefix}command[{i}]", command_text))
    print("\n".join(lines))


class MpProcessGroup:
    def __init__(self, procs):
        self.procs = procs

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
                # Can hang here.
                proc.join()


def main():
    count = 5
    duration_sec = 1.0

    pub_ready = mp.Value(ctypes.c_int)
    pub_ready.value = 0
    sub_ready = mp.Value(ctypes.c_int)
    sub_ready.value = 0

    procs = MpProcessGroup([
        mp.Process(target=wrap_rclpy, args=[pub_main, count, pub_ready]),
        mp.Process(target=wrap_rclpy, args=[sub_main, count, sub_ready]),
    ])
    procs.start()

    while True:
        if pub_ready.value != 0 and sub_ready.value != 0:
            break
        time.sleep(0.001)

    print(f"Pub + sub ready. Stopping after {duration_sec}")
    print()

    t_end = time.time() + duration_sec

    try:
        while time.time() < t_end:
            procs.poll()
            time.sleep(1 / RATE_HZ)
    except KeyboardInterrupt:
        pass
    finally:
        procs.close()


if __name__ == "__main__":
    main()
