import argparse
from functools import partial
from subprocess import Popen, PIPE, STDOUT
import sys
import time

import six

from process_util import on_context_exit, read_available, signal_processes


def main():
    args = ["stdbuf", "--output=0", sys.executable, "sub_level.py"]
    if six.PY3:
        simple_encoding = dict(encoding="utf8", universal_newlines=True, bufsize=1)
    else:
        simple_encoding = dict()

    t_start = time.time()
    t_max = 2.
    count = 0

    lines = []
    lines_expected = ["0", "1", "2", "3", "4"]

    proc = Popen(args, stdout=PIPE, stderr=STDOUT, **simple_encoding)
    with on_context_exit(partial(signal_processes, [proc])):
        while True:
            assert proc.poll() is None
            line = read_available(proc.stdout).strip()
            if line:
                print("sub: {}".format(line))
                lines.append(line)
            if lines == lines_expected:
                print("Success")
                break
            t = time.time() - t_start
            if t >= t_max:
                print("Timeout")
                sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
