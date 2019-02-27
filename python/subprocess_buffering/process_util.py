from contextlib import contextmanager
import os
import select
import signal

@contextmanager
def on_context_exit(f):
    try:
        yield
    finally:
        f()


def read_available(f, timeout=0.0, chunk_size=1024, empty=None):
    """
    Reads all available data on a given file. Useful for using PIPE with Popen.

    @param timeout Timeout for `select`.
    @param chunk_size How much to try and read.
    @param empty Starting point / empty value. Default value is empty byte
    array.
    """
    readable, _, _ = select.select([f], [], [f], timeout)
    if empty is None:
        empty = bytes()
    out = empty
    if f in readable:
        while True:
            cur = os.read(f.fileno(), chunk_size)
            out += cur
            if len(cur) < chunk_size:
                break
    return out


def signal_processes(process_list, sig=signal.SIGINT, block=True):
    """
    Robustly sends a singal to processes that are still alive. Ignores status
    codes.

    @param process_list List[Popen] Processes to ensure are sent a signal.
    @param sig Signal to send. Default is `SIGINT1.
    @param block Block until process exits.
    """
    for process in process_list:
        if process.poll() is None:
            process.send_signal(sig)
    if block:
        for process in process_list:
            if process.poll() is None:
                process.wait()
