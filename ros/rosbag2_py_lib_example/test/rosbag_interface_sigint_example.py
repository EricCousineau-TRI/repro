"""
Simple example that publishes indefinitely until Ctrl+C (SIGINT).

If we have control of Ctrl+C, then we should be able to catch this in Python
and disable logging.
If we don't have control (the nominal case), then we will have no control.
"""

import argparse
import time

import rclpy
import std_msgs.msg

from example.ros.cc import RclcppInit
from example.rosbag_interface import Recorder


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bag_path", type=str)
    args = parser.parse_args()

    rclpy.init()
    RclcppInit()

    recorder = Recorder()
    recorder.start(args.bag_path)

    node = rclpy.create_node("example_node")
    publisher = node.create_publisher(std_msgs.msg.String, "/test", 1)

    rate = LoopRate(100.0)
    try:
        while True:
            msg = std_msgs.msg.String()
            msg.data = str(time.time())
            publisher.publish(msg)
            rate.sleep()
    except KeyboardInterrupt:
        pass
    finally:
        recorder.stop()


if __name__ == "__main__":
    main()
