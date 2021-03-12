#!/usr/bin/env python3

import os
from os.path import abspath, dirname, isfile, isdir
from subprocess import PIPE, run
import sys


class UserError(RuntimeError):
    pass


def eprint(s):
    print(s, file=sys.stderr)


def shell(cmd, check=True):
    """Executes a shell command."""
    eprint(f"+ {cmd}")
    return run(cmd, shell=True, check=check)


def cd(p):
    eprint(f"+ cd {p}")
    os.chdir(p)


def mkdir(p):
    eprint(f"+ mkdir {p}")
    os.mkdir(p)


def mkcd(p):
    mkdir(p)
    cd(p)


def parent_dir(p, *, count):
    for _ in range(count):
        p = dirname(p)
    return p


def main():
    assert sys.version_info[:2] == (3, 6)

    source_tree = parent_dir(abspath(__file__), count=1)

    # Download mesh.
    cd(source_tree)
    commit = "c8c27c15"  # branch: melodic-devel
    if not isfile("base.dae"):
        shell(f"wget https://raw.githubusercontent.com/ros-industrial/universal_robot/{commit}/ur_description/meshes/ur10/visual/base.dae")

    # Run w/ different versions.
    cd(source_tree)
    print("[ System Version ]")
    shell("python3 ./repro.py", check=False)


if __name__ == "__main__":
    try:
        main()
    except UserError as e:
        eprint(e)
        sys.exit(1)
