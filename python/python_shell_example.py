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


@contextmanager
def safe_git_restore_context():
    # WARNING: This expects that your `gitignore` ignores a sufficient amount
    # of stuff to not get thrown off.
    # Ensure that there are no changes in Git that we will lose.
    status_text = subshell("git --no-pager status -s")
    if status_text != "":
        raise UserError(f"Dirty tree! Cannot proceed\n{status_text}")
    starting_ref = subshell("git rev-parse --abbrev-ref HEAD")
    if starting_ref == "HEAD":
        starting_ref = subshell("git rev-parse HEAD")
    eprint(f"Starting git ref: {starting_ref}")
    shell("git log -n 1 --oneline --no-decorate")
    try:
        yield starting_ref
    finally:
        # It is safe to do this since we've prevent usage of this script in a
        # dirty workspace.
        eprint(f"Returning to git ref: {starting_ref}")
        shell(f"git checkout -f {starting_ref}")


def do_some_things():
    raise NotImplementError()


def do_more_things():
    raise NotImplementError()


def main():
    source_tree = parent_dir(abspath(__file__), count=3)
    cd(source_tree)

    with safe_git_restore_context() as starting_ref:
        do_some_things()

    do_more_things()


if __name__ == "__main__":
    try:
        main()
    except UserError as e:
        eprint(e)
        sys.exit(1)
