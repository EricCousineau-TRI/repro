"""
Tests (and examples) for `process_util`.
"""

from contextlib import closing
from functools import partial
from subprocess import Popen
import time
import unittest

from process_util import (
    CapturedProcess,
    CapturedProcessGroup,
    on_context_exit,
    print_prefixed,
    signal_processes,
)


class Test(unittest.TestCase):
    def test_captured_process(self):
        # Signaling.
        p = Popen(["sleep", "10"])
        with on_context_exit(lambda: signal_processes([p])):
            time.sleep(1.0)
        assert p.returncode == -2

        # Pipe I/O.
        args = ["bash", "-c", "while :; do echo Hello; read; done"]
        p = CapturedProcess(
            args,
            on_new_text=partial(print_prefixed, prefix=" | "),
            simple_encoding=True,
        )
        with p.scope:
            for i in range(3):
                while "Hello" not in p.output.get_text():
                    time.sleep(0.01)
                    p.poll()
                p.output.clear()
                p.proc.stdin.write("\n")

    def _run_captured_process_group(self, *args, **kwargs):
        procs = CapturedProcessGroup()
        with closing(procs):
            procs.add(*args, **kwargs)
            t_next = time.time() + 0.5
            while time.time() < t_next:
                assert procs.poll() == {}
                time.sleep(0.1)
            print("Exiting")
        print("  Done")

    def test_captured_process_group_shell(self):
        self._run_captured_process_group(
            "sleep", ["sleep", "1d"], shell=False, verbose=True
        )
        # Why does this hang?!!!
        self._run_captured_process_group(
            "sleep", "sleep 1d", shell=True, verbose=True
        )


if __name__ == "__main__":
    unittest.main()
