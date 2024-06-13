"""
Custom rosbag recording interface.

This is done so that we minimize the time needed to begin recording when
launching from an existing process.
"""

import os
import shutil

from rosbag2_py import StorageOptions

# ros2bag_py.Recorder.start() will mess with signal handlers. For more
# details, see https://github.com/ros2/rosbag2/issues/1678
# So we use own binding recorder
from example import rosbag_recording


def _delete_path(path):
    if os.path.exists(path):
        if os.path.isfile(path):
            os.remove(path)
            print(f"File {path} deleted.")
        elif os.path.isdir(path):
            shutil.rmtree(path)
            print(f"Directory {path} and its contents deleted.")


class Recorder:
    # TODO(masayuki): allow other options
    def __init__(
        self, storage_id="mcap", rmw_serialization_format="cdr", regex=""
    ):
        self._recorder = rosbag_recording.Recorder()
        self._storage_id = storage_id
        self._rmw_serialization_format = rmw_serialization_format
        self._regex = regex

        self._output = None
        self.num = 0

    def start(self, output="/tmp/test.bag"):
        assert output is not None
        print(f"start rosbag recording. save path: {output}")

        if os.path.exists(output):
            raise RuntimeError(f"rosbag path already exists! {output}")
        self._output = output

        storage_options = StorageOptions(
            uri=output,
            storage_id=self._storage_id,
        )
        record_options = rosbag_recording.RecordOptions()
        record_options.all = True
        record_options.exclude = self._regex
        record_options.rmw_serialization_format = (
            self._rmw_serialization_format
        )
        self._recorder.record(storage_options, record_options)

    def stop(self):
        print(f"stop rosbag recording. file will be saved at {self._output}")
        self._recorder.cancel()

    def cancel(self):
        print("cancel rosbag recording.")
        self._recorder.cancel()
        _delete_path(self._output)

    def is_recording(self):
        return self._recorder.is_recording()
