"""
Isolated set of definitions to add in basic shell scripting via Python.

Goals:
- Make it easy to port shell code to Python.
- Ensure job control isn't that complicated.

Non-goals:
- Make it super Pythonic.

Derived from `repro` stuff.
"""

from contextlib import contextmanager
import os
import shlex
import signal
from subprocess import Popen, CompletedProcess, run, PIPE
import sys
from textwrap import dedent
import time

sys.dont_write_bytecode = True

SUBSHELL_VERBOSE = False
SHELL_EXPAND_VERBOSE = False

# Section: Shell scripting.


class UserError(RuntimeError):
    pass


def shlex_join(argv):
    # TODO(eric.cousineau): Replace this with `shlex.join` when we exclusively
    # use Python>=3.8.
    return " ".join(map(shlex.quote, argv))


def eprint(s):
    print(s, file=sys.stderr)


def _signal_process(process, sig=signal.SIGINT, block=True, close_streams=True):
    if process.poll() is None:
        process.send_signal(sig)
    if close_streams:
        for stream in [process.stdin, process.stdout, process.stderr]:
            if stream is not None and not stream.closed:
                stream.close()
    if block:
        if process.poll() is None:
            process.wait()


@contextmanager
def _intercept_sigint_for_proc(proc):
    # https://github.com/amoffat/sh/issues/495#issuecomment-801069064
    try:
        yield
    except KeyboardInterrupt:
        while True:
            try:
                _signal_process(proc)
                break
            except KeyboardInterrupt:
                print("(Catching interrupt again...")
        raise


def shell(args, shell=True, check=True, **kwargs):
    """Executes a shell command."""
    if shell:
        assert isinstance(args, str)
        args = dedent(args)
        cmd = args
        if SHELL_EXPAND_VERBOSE:
            cmd = f"set -x; {cmd}"
    else:
        cmd = shlex_join(args)
    eprint(f"+ {cmd.strip()}")
    proc = Popen(args, shell=shell, **kwargs)
    with _intercept_sigint_for_proc(proc):
        proc.wait()
    valid_returncodes = {0}
    if check and proc.returncode not in valid_returncodes:
        raise UserError(f"Process failed with code {proc.returncode}: {cmd}")
    return CompletedProcess(args=args, returncode=proc.returncode)


def subshell(cmd, check=True, stderr=None, strip=True, **kwargs):
    """Executs a subshell in a capture."""
    if SUBSHELL_VERBOSE:
        eprint(f"+ $({cmd.strip()})")
    if SHELL_EXPAND_VERBOSE:
        cmd = f"set -x; {cmd}"
    result = run(cmd, shell=True, stdout=PIPE, stderr=stderr, encoding="utf8", **kwargs)
    if result.returncode != 0 and check:
        if stderr == PIPE:
            eprint(result.stderr)
        eprint(result.stdout)
        raise UserError(f"Exit code {result.returncode}: {cmd}")
    out = result.stdout
    if strip:
        out = out.strip()
    return out
