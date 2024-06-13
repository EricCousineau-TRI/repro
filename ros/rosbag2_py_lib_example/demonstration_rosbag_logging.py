"""
This script contains a class for managing rosbag record used in demonstration.
"""

import atexit
import signal
import subprocess

from example.rosbag_interface import Recorder


class RecorderWithPolling:
    def __init__(self, recorder, proc=None):
        self._recorder = recorder
        self._proc = proc

    def start(self, output):
        self._recorder.start(output)

    def stop(self):
        self._recorder.stop()

    def cancel(self):
        self._recorder.cancel()

    def poll(self):
        assert self._recorder.is_recording()
        if self._proc is not None:
            ret_code = self._proc.poll()
            assert ret_code is None, ret_code


class RosbagConfig:
    def __init__(
        self,
        storage_id: str = "mcap",
        regex: str = "",
    ):
        self._storage_id = storage_id
        self._regex = regex

    def create(self):
        recorder = Recorder(storage_id=self._storage_id, regex=self._regex)
        return RecorderWithPolling(recorder)
