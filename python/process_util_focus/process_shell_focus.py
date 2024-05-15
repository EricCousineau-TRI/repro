import os
import signal
from subprocess import PIPE, Popen, STDOUT
import time


def maybe_get_new_session_pgid(process):
    """
    Returns new session pgid, or None if it's the same session as this process."""
    my_pgid = os.getpid()
    process_pgid = os.getpgid(process.pid)
    if process_pgid != my_pgid:
        return process_pgid
    else:
        return None


def signal_process(process, sig=signal.SIGINT, block=True, close_streams=True):
    if process.poll() is None:
        # Use suggestion in https://github.com/python/cpython/issues/119059
        process_pgid = maybe_get_new_session_pgid(process)
        if process_pgid is not None:
            os.killpg(process_pgid, sig)
        else:
            process.send_signal(sig)
    if close_streams:
        for stream in [process.stdin, process.stdout, process.stderr]:
            if stream is not None and not stream.closed:
                stream.close()
    if block:
        if process.poll() is None:
            process.wait()


def run_captured_process(args, *, shell, sig, start_new_session=False):
    # Start a process, wait a brief amount of time, then try to exit.
    proc = Popen(args, shell=shell, start_new_session=start_new_session)
    try:
        t_next = time.time() + 0.5
        while time.time() < t_next:
            assert proc.poll() is None
            time.sleep(0.1)
        print("Exiting")
    finally:
        signal_process(proc, sig=sig)
    print("  Done")


def main():
    # Does not hang.
    run_captured_process(["sleep", "1d"], shell=False, sig=signal.SIGINT)
    # Does not hang.
    run_captured_process("sleep 1d", shell=True, sig=signal.SIGABRT)
    # Does not hang - fixed!
    run_captured_process(
        "sleep 1d", shell=True, start_new_session=True, sig=signal.SIGINT
    )


if __name__ == "__main__":
    main()
