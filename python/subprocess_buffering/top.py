import argparse
from functools import partial
from subprocess import Popen, PIPE, STDOUT
import sys
import time

import six

from process_util import on_context_exit, read_available, signal_processes


def read_available_text(f):
    line = read_available(f)
    if isinstance(line, bytes):
        line = line.decode("utf8")
    if six.PY2:
        line = str(line)
    return line.strip()


def main():
    assert sys.argv[1] == "--"
    args = sys.argv[2:]
    print(args)
    if six.PY3:
        simple_encoding = dict(encoding="utf8", universal_newlines=True, bufsize=1)
    else:
        simple_encoding = dict()

    t_start = time.time()
    t_max = 0.3
    count = 0

    lines = []
    lines_expected = ["0", "1", "2"]

    proc = Popen(args, stdout=PIPE, stderr=STDOUT, **simple_encoding)
    with on_context_exit(partial(signal_processes, [proc])):
        while True:
            assert proc.poll() is None
            line = read_available_text(proc.stdout)
            if line:
                lines.append(line)
            if lines == lines_expected:
                print("Success: {}".format(lines))
                break
            t = time.time() - t_start
            if t >= t_max:
                print("ERROR: Timeout")
                sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
