"""
Tools for targeted profiling of Python application code.
"""

from contextlib import contextmanager
import cProfile as profile
import os
import pstats
import shlex
import signal
import subprocess
import time

_PTRACE_NEEDED = """
ptrace is currently not enabled and profiling will not work without sudo.
Please temporarily enable it using the following:

    echo "0" | sudo tee /proc/sys/kernel/yama/ptrace_scope

For more details, see:
https://github.com/benfred/py-spy/tree/v0.3.14?tab=readme-ov-file#when-do-you-need-to-run-as-sudo

"""  # noqa


def _fail_fast_if_ptrace_scope_not_enabled():
    ptrace_file = "/proc/sys/kernel/yama/ptrace_scope"
    with open(ptrace_file, "r") as f:
        text = f.read().strip()
    if text != "0":
        raise RuntimeError(_PTRACE_NEEDED)


@contextmanager
def use_py_spy(
    output_file=None,
    *,
    # py_spy_args=["-r", "25", "--native"],  # native, but slows down
    py_spy_args=["-r", "100", "--nonblocking", "--threads"],  # pure python
    sudo=False,
    sleep_before=0.0,
):
    """
    Sample-based profiling for Python code that also handles CPython bindings
    (e.g. pydrake, etc.) for a targeted section of code.

    As an example:

        some_code_i_dont_want_to_profile()
        with use_py_spy():
            my_long_function_im_interested_in()
        some_other_code_i_dont_want_to_profile()

    This works by using `py-spy --pid {this_process}`, so that we only get
    profiling on a targeted section of code.

    If you are interested / able to profile the entirety of a process, you
    should launch said process using `py-spy python {script}`. However, if you
    need more targeted profiling, use this context.

    Arguments:
        output_file: Output SVG file. To view the results, this output SVG file
            in your webbrowser. If None, will default to
            "~/py_spy_profile.svg".
        py_spy_args: Arguments to pass to py-spy for profiling. The defaults
            were found to be useful for getting relatively accurate sampling
            in `python_profiling_manual_test`.
        sudo: If True, will run `py-spy` under `sudo`, but will require
            authentication. If False, will require ptrace to be enabled.
            See documents in _PTRACE_NEEDED variable.
        sleep_before: If non-zero, will sleep for this number of seconds before
            resuming execution. This may be necessary if py-spy has
            non-negligible delay before it has attached and started profiling.

    For more details, please see README for:
    https://github.com/benfred/py-spy

    For motivation, see:
    https://stackoverflow.com/questions/74201118/pydrake-how-do-i-identify-slow-python-leafsystems-to-possibly-rewrite-in-c/74201119#74201119
    """  # noqa
    if not sudo:
        _fail_fast_if_ptrace_scope_not_enabled()
    if output_file is None:
        output_file = os.path.expanduser("~/py_spy_profile.svg")
    bin = "py-spy"
    args = [
        bin,
        "record",
        "-o",
        output_file,
        "--pid",
        str(os.getpid()),
    ] + py_spy_args
    if sudo:
        args = ["sudo"] + args

    args_text = shlex.join(args)
    print(f"Profiling with: {args_text}")
    proc = subprocess.Popen(args)

    if sleep_before > 0:
        print(f"  Sleeping for {sleep_before}s before doing work")
        time.sleep(sleep_before)
        print("  Done sleeping; doing work to be profiled")

    t_start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - t_start
        print(f"Elapsed time: {elapsed} sec")
        print(f"View results using:")
        print(f"  x-www-browser {output_file}")
        proc.send_signal(signal.SIGINT)
        proc.wait()


@contextmanager
def use_python_profile(output_file=None):
    """
    Profiling with Python's builtin `profile` module.

    Arguments:
        output_file: Output text file. If None, will default to
            "~/python_profile_stats.txt".
    """
    if output_file is None:
        output_file = os.path.expanduser("~/python_profile_stats.txt")
    pr = profile.Profile()
    pr.enable()
    t_start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - t_start
        print(f"Elapsed time: {elapsed} sec")
        pr.disable()
        stats = pstats.Stats(pr)
        stats.sort_stats("tottime", "cumtime")
        stats.dump_stats(output_file)
        print(f"View results using:")
        print(f"  ./run //tools:tuna {output_file}")
