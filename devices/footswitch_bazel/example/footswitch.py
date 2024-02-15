"""
FootSwitch driver using keyboard redirection.
Avoids using pygame because managing windows can be painful.

For product info, setup instructions, see
//tools/workspace/footswitch/README.md.
"""

import copy
from enum import Enum
import threading

import hid


class Pedal(Enum):
    Left = "Left"
    Middle = "Middle"
    Right = "Right"


# These are the keys configured per `//tools/workspace/footswitch:README`.
# For key mapping, see:
# https://github.com/rgerganov/footswitch/blob/7c448d24/common.c#L35-L314
DEFAULT_KEY_MAP = {
    # Enum: (hid Key)
    Pedal.Left: 0x4B,
    Pedal.Middle: 0x51,
    Pedal.Right: 0x4E,
}


class NoFootswitchDetected(Exception):
    def __init__(self):
        super().__init__("No footswitch devices detected")


class _HidPoll:
    """
    Polling using libhidapi. This does not require window focus (from either
    pygame or OpenCV).

    Note that this device can only be opened in one process at a time.
    """

    def __init__(
        self,
        *,
        # PCSensor Footpedal
        vendor_product_ids=(
            (0x1A86, 0xE026),
            (0x3553, 0xB001),
        ),
    ):
        self.vendor_product_ids = vendor_product_ids

        self._thread_running = True
        self._thread_error = None
        self._thread = None

        self._lock = threading.Lock()
        self._keys = None

        self._start_thread()

    def __del__(self):
        self._thread_running = False
        if self._thread is not None:
            self._thread.join()

    def _assert_running(self):
        if self._thread_error is not None:
            e = self._thread_error
            self._thread_error = None
            raise e
        if not self._thread_running:
            raise RuntimeError("Hid polling not running?")

    def _start_thread(self):
        self._thread = threading.Thread(target=self._run)
        self._thread.daemon = True
        self._thread.start()
        # Wait for first read.
        while True:
            with self._lock:
                self._assert_running()
                if self._keys is not None:
                    break

    def get_pressed_keys(self):
        self._assert_running()
        with self._lock:
            keys = copy.copy(self._keys)
        return keys

    @staticmethod
    def _read_hid_pressed_keys(data):
        if len(data) == 0:
            return None
        # Observed from printout out the raw data itself.
        assert data[0] == 1
        assert data[-1] == 0
        keys = set()
        for d in data[1:-1]:
            if d == 0:
                continue
            keys.add(d)
        return keys

    def _open_device(self):
        device = hid.device()
        for vendor_id, product_id in self.vendor_product_ids:
            try:
                device.open(vendor_id, product_id)
                return device
            except OSError as e:
                if "open failed" in str(e):
                    continue
                else:
                    raise
        raise NoFootswitchDetected()

    def _run(self):
        try:
            device = self._open_device()
            # Using this size because that's what is used elsewhere:
            # https://github.com/rgerganov/footswitch/blob/7c448d2/footswitch.c#L241
            read_size = 8
            while self._thread_running:
                data = device.read(read_size, timeout_ms=10)
                keys = self._read_hid_pressed_keys(data)
                with self._lock:
                    if self._keys is None:
                        self._keys = set()
                    if keys is not None:
                        self._keys = keys

        except BaseException as e:
            self._thread_running = False
            self._thread_error = e


class FootSwitch:
    def __init__(
        self,
        key_map=DEFAULT_KEY_MAP,
    ):
        self.key_map = key_map
        self._hid = _HidPoll()

    def get_events(self):
        events = dict()
        hid_keys = self._hid.get_pressed_keys()
        for pedal, hid_key in self.key_map.items():
            value = hid_key in hid_keys
            events[pedal] = value
        return events

    @staticmethod
    def are_any_pressed(events):
        for value in events.values():
            if value:
                return True
        return False
