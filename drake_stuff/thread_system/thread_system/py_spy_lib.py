"""
https://stackoverflow.com/questions/74201118/pydrake-how-do-i-identify-slow-python-leafsystems-to-possibly-rewrite-in-c/74201119#74201119
https://github.com/EricCousineau-TRI/repro/tree/6048da3/drake_stuff/python_profiling

echo "0" | sudo tee /proc/sys/kernel/yama/ptrace_scope

WARNING: This seems to be way slower on Jammy machine?! It incurs a 4x slow
down :(
"""

from contextlib import contextmanager
import os
import signal
import subprocess
import time


@contextmanager
def use_py_spy(output_file, *, sudo=False):
    """Use py-spy in specific context."""
    args = [
        "py-spy",
        "record",
        # Can be slow.
        # "--native",
        "-o",
        output_file,
        "--pid",
        str(os.getpid()),
    ]
    if sudo:
        args = ["sudo"] + args
    p = subprocess.Popen(args)
    # TODO(eric.cousineau): Startup time of py-spy may lag behind other
    # instructions. However, maybe can assume this isn't critical for profiling
    # purposes?
    time.sleep(0.1)
    t_start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - t_start
        print(f"Elapsed time: {elapsed} sec")
        p.send_signal(signal.SIGINT)
        p.wait()
