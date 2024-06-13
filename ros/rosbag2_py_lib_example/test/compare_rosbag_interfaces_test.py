"""
This is test for comparing ros2bag record verb and own interface.
"""

import atexit
import os
import shutil
import time
import unittest
import weakref

import rclpy
from rclpy.serialization import deserialize_message
import rosbag2_py
from rosidl_runtime_py.utilities import get_message
from std_msgs.msg import String

from example.process_util import (
    CapturedProcessGroup,
    wait_for_process_with_condition,
)
from example.ros.ros_operations import activate_ros_isolation
from example.ros.cc import RclcppInit
from example.runfiles import Rlocation
from example.demonstration_rosbag_logging import RosbagConfig


class RecorderUsingProcess:
    def __init__(self, block=True):
        self._procs = CapturedProcessGroup()
        self._block = block

        def _stop_procs(procs_ref):
            procs = procs_ref()
            if procs is not None:
                procs.close()

        atexit.register(_stop_procs, weakref.ref(self._procs))

    def start(self, path):
        proc = self._procs.add(
            "test_record",
            [
                Rlocation("example/tools/ros2"),
                "bag",
                "record",
                "-a",
                "-s",
                "mcap",
                "-o",
                path,
            ],
        )

        if self._block:
            # wait for rosbag started
            assert wait_for_process_with_condition(
                proc,
                lambda: "Recording..." in proc.output.get_text(),
            )

    def stop(self):
        self._procs.remove("test_record", close=True, block=True)


class RecorderUsingOwn:
    def __init__(self):
        rosbag_config = RosbagConfig()
        self._recorder = rosbag_config.create()

    def start(self, path):
        self._recorder.start(path)

    def stop(self):
        self._recorder.stop()


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


class Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_count = 0

    def setUp(self):
        self._tmp_dir = os.environ.get("TEST_TMPDIR", "/tmp")
        self._bag_path = os.path.join(self._tmp_dir, "test_bag")
        self._node = rclpy.create_node("test_node")
        self._publisher = self._node.create_publisher(
            String,
            "/test",
            1,
        )
        self._published_msg_list = []

    def tearDown(self):
        self._node.destroy_node()
        if os.path.isdir(self._bag_path):
            shutil.rmtree(self._bag_path)

    def _load_rosbag_messages(self):
        storage_options = rosbag2_py.StorageOptions(
            uri=self._bag_path, storage_id="mcap"
        )
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        )
        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)
        topic_name_to_type = {
            topic_and_type.name: get_message(topic_and_type.type)
            for topic_and_type in reader.get_all_topics_and_types()
        }

        receive_msg_list = []
        while reader.has_next():
            topic_name, data, timestamp = reader.read_next()
            msg_type = topic_name_to_type[topic_name]
            if topic_name in ["/rosout", "/parameter_events"]:
                # Skip log messages.
                continue
            self.assertEqual(msg_type, String)
            msg = deserialize_message(data, msg_type)
            receive_msg_list.append(msg)
        return receive_msg_list

    def _check_rosbag(self):
        if os.path.exists(self._bag_path):
            receive_msg_list = self._load_rosbag_messages()
        else:
            receive_msg_list = []

        num_published = len(self._published_msg_list)
        self.assertGreater(num_published, 0)

        receive_msg_list.reverse()
        for msg in receive_msg_list:
            pub_msg = self._published_msg_list.pop()
            self.assertEqual(pub_msg.data, msg.data)

        all_received = False
        if len(self._published_msg_list) == 0:
            all_received = True
            print(f"All {num_published} messages are received.")
        else:
            num_lost = len(self._published_msg_list)
            percent_lost = 100 * num_lost / num_published
            print(
                f"{num_lost} / {num_published} ({percent_lost}%) messages "
                f"are lost."
            )
            if len(receive_msg_list) > 0:
                first_record_time = float(receive_msg_list[-1].data)
                first_publish_time = float(self._published_msg_list[0].data)
                delta_time = first_record_time - first_publish_time
                print(f"first_record_time - first_publish_time: {delta_time}")

        return all_received

    def _record_and_publish(self, recorder, duration=3.0):
        time_before_start = time.time()
        recorder.start(self._bag_path)
        time_to_start = time.time() - time_before_start

        # Publish messages.
        t_final = self._node.get_clock().now() + rclpy.duration.Duration(
            seconds=duration
        )
        # N.B. Use custom loop-rate to avoid needing asynchronous spinning
        # thread.
        rate = LoopRate(hz=100.0)
        while rclpy.ok() and self._node.get_clock().now() < t_final:
            msg = String()
            msg.data = str(time.time())
            self._published_msg_list.append(msg)
            self._publisher.publish(msg)
            rate.sleep()

        time_before_stop = time.time()
        recorder.stop()
        time_to_stop = time.time() - time_before_stop

        print(f"time needed to start: {time_to_start}")
        print(f"time needed to stop: {time_to_stop}")

    def test_record_using_process_blocked(self):
        self._record_and_publish(RecorderUsingProcess(block=True))
        self._check_rosbag()
        # N.B. Depending on the machine load, we may not receive all messages
        # in this case - most likely, the time from seeing the message printed
        # to when it actually starts recording may be different.

    def test_record_using_process_not_blocked(self):
        self._record_and_publish(
            RecorderUsingProcess(block=False),
            # Because of the slow startup time, we need to record longer than
            # the normal amount.
            duration=5.0,
        )
        # This will most likely lose data lose since we do not wait until
        # recording is ready.
        self._check_rosbag()

    def test_record_using_own_interface(self):
        self._record_and_publish(RecorderUsingOwn())
        all_received = self._check_rosbag()
        self.assertTrue(all_received)


if __name__ == "__main__":
    activate_ros_isolation()
    rclpy.init()
    RclcppInit()
    unittest.main()
