from contextlib import closing
import os
from pathlib import Path
import shutil
import threading
import unittest

import rclpy
from std_msgs.msg import String

from example.process_util import (
    CapturedProcessGroup,
    wait_for_process_with_condition,
)
from example.ros.ros_operations import activate_ros_isolation
from example.runfiles import Rlocation


class Test(unittest.TestCase):
    def setUp(self):
        rclpy.init()
        self.tmp_dir = os.environ.get("TEST_TMPDIR", "/tmp")
        self.bag_dir = Path(self.tmp_dir) / "test_bag"
        if self.bag_dir.exists():
            shutil.rmtree(self.bag_dir)

    def teadDown(self):
        rclpy.shutdown()

    def test_sigint(self):
        procs = CapturedProcessGroup()
        example_bin = Rlocation(
            "example/rosbag_interface_sigint_example"
        )
        example = procs.add("example", [example_bin, self.bag_dir])

        node = rclpy.create_node("publish_checker")
        is_running = True

        def spin_node():
            while is_running and rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.001)

        th = threading.Thread(target=spin_node, daemon=True)
        th.start()

        self.subscribed = False

        def callback(msg):
            assert type(msg) is String
            self.subscribed = True

        subscriber = node.create_subscription(String, "/test", callback, 1)

        with closing(procs):
            # ensure proc be in try-except section
            wait_for_process_with_condition(procs, lambda: self.subscribed)

            wait_for_process_with_condition(procs, timeout=1.0)
        self.assertEqual(procs.poll(), {"example": 0})
        self.assertTrue(os.path.exists(self.bag_dir))

        is_running = False
        th.join()


if __name__ == "__main__":
    activate_ros_isolation()
    unittest.main()
