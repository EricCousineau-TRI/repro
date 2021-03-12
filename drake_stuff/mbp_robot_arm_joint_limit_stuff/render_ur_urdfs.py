#!/usr/bin/env python3

#!/usr/bin/env python3

"""
Example of doing cmdline script-y things in Python (rather than a bash script).

Derived from:
https://github.com/RobotLocomotion/drake/blob/7de24898/tmp/benchmark/generate_benchmark_from_master.py
"""

from contextlib import contextmanager
import os
from os.path import abspath, dirname, isdir
import shlex
from subprocess import run, PIPE
import signal
from textwrap import dedent
import sys


class UserError(RuntimeError):
    pass


def shlex_join(argv):
    # TODO(eric.cousineau): Replace this with `shlex.join` when we exclusively
    # use Python>=3.8.
    return " ".join(map(shlex.quote, argv))

def eprint(s):
    print(s, file=sys.stderr)


def shell(cmd, check=True):
    """Executes a shell command."""
    eprint(f"+ {cmd}")
    return run(cmd, shell=True, check=check)


# def subshell(cmd, check=True, stderr=None, strip=True):
#     """Executs a subshell in a capture."""
#     eprint(f"+ $({cmd})")
#     result = run(cmd, shell=True, stdout=PIPE, stderr=stderr, encoding="utf8")
#     if result.returncode != 0 and check:
#         if stderr == PIPE:
#             eprint(result.stderr)
#         eprint(result.stdout)
#         raise UserError(f"Exit code {result.returncode}: {cmd}")
#     out = result.stdout
#     if strip:
#         out = out.strip()
#     return out


def cd(p):
    eprint(f"+ cd {p}")
    os.chdir(p)


def parent_dir(p, *, count):
    for _ in range(count):
        p = dirname(p)
    return p


@contextmanager
def pushd(p):
    cur_dir = os.getcwd()
    cd(p)
    yield
    cd(cur_dir)


def signal_processes(process_list, sig=signal.SIGINT, block=True, close_streams=True):
    """
    Robustly sends a singal to processes that are still alive. Ignores status
    codes.
    """
    for process in process_list:
        if process.poll() is None:
            process.send_signal(sig)
        if close_streams:
            for stream in [process.stdin, process.stdout, process.stderr]:
                if stream is not None and not stream.closed:
                    stream.close()
    if block:
        for process in process_list:
            if process.poll() is None:
                process.wait()


def do_some_things():
    raise NotImplementError()


def do_more_things():
    raise NotImplementError()


def main():
    source_tree = parent_dir(abspath(__file__), count=1)
    cd(source_tree)

    if "ROS_DISTRO" not in os.environ:
        raise UserError("Please run under `./ros_setup.bash`, or whatevs")

    cd("repos/universal_robot/ur_description")
    # Use URI that is unlikely to be used.
    os.environ["ROS_MASTER_URI"] = "http://localhost:11321"

    # Start a roscore, 'cause blech.
    roscore = subprocess.Popen(
        ["roscore"],
        stdout=PIPE,
        stderr=STDOUT,
    )
    try:
        # Blech.
        time.sleep(1.0)

        shell("")

    finally:
        signal_processes([roscore])


if __name__ == "__main__":
    try:
        main()
    except UserError as e:
        eprint(e)
        sys.exit(1)
