#!/usr/bin/env python3

"""
Example of doing cmdline script-y things in Python (rather than a bash script).

Derived from:
https://github.com/RobotLocomotion/drake/blob/7de24898/tmp/benchmark/generate_benchmark_from_master.py
"""

from contextlib import closing
import os
from os.path import abspath, dirname
from subprocess import PIPE, run
import sys
from textwrap import indent

from process_util import CapturedProcess, bind_print_prefixed
import yaml


class UserError(RuntimeError):
    pass


def eprint(s):
    print(s, file=sys.stderr)


def shell(cmd, check=True):
    """Executes a shell command."""
    eprint(f"+ {cmd}")
    return run(cmd, shell=True, check=check)


def subshell(cmd, check=True, stderr=None, strip=True):
    """Executs a subshell in a capture."""
    eprint(f"+ $({cmd})")
    result = run(cmd, shell=True, stdout=PIPE, stderr=stderr, encoding="utf8")
    if result.returncode != 0 and check:
        if stderr == PIPE:
            eprint(result.stderr)
        eprint(result.stdout)
        raise UserError(f"Exit code {result.returncode}: {cmd}")
    out = result.stdout
    if strip:
        out = out.strip()
    return out


def cd(p):
    eprint(f"+ cd {p}")
    os.chdir(p)


def parent_dir(p, *, count):
    for _ in range(count):
        p = dirname(p)
    return p


FLAVORS = [
    "ur3",
    "ur3e",
    "ur5",
    "ur5e",
]


def main():
    source_tree = parent_dir(abspath(__file__), count=1)
    cd(source_tree)

    if "ROS_DISTRO" not in os.environ:
        raise UserError("Please run under `./ros_setup.bash`, or whatevs")

    cd("repos/universal_robot")
    # Use URI that is unlikely to be used.
    os.environ["ROS_MASTER_URI"] = "http://localhost:11321"
    os.environ[
        "ROS_PACKAGE_PATH"
    ] = f"{os.getcwd()}:{os.environ['ROS_PACKAGE_PATH']}"

    cd("ur_description")

    urdf_files = []

    # Start a roscore, 'cause blech.
    roscore = CapturedProcess(
        ["roscore", "-p", "11321"],
        on_new_text=bind_print_prefixed("[roscore] "),
    )
    with closing(roscore):
        # Blech.
        while "started core service" not in roscore.output.get_text():
            assert roscore.poll() is None

        for flavor in FLAVORS:
            shell(f"roslaunch ur_description load_{flavor}.launch")
            urdf_file = f"urdf/{flavor}.urdf"
            output = subshell(f"rosparam get /robot_description")
            # Blech :(
            content = yaml.load(output)
            with open(urdf_file, "w") as f:
                f.write(content)
            urdf_files.append(urdf_file)

        print("\n\n")
        print("Generated URDF files:")
        print(indent("\n".join(urdf_files), "  "))


if __name__ == "__main__":
    try:
        main()
        print("[ Done ]")
    except UserError as e:
        eprint(e)
        sys.exit(1)
