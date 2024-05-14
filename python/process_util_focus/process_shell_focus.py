import signal
from subprocess import PIPE, Popen, STDOUT
import time


def signal_process(process, sig=signal.SIGINT, block=True, close_streams=True):
    if process.poll() is None:
        process.send_signal(sig)
    if close_streams:
        for stream in [process.stdin, process.stdout, process.stderr]:
            if stream is not None and not stream.closed:
                stream.close()
    if block:
        if process.poll() is None:
            process.wait()


def run_captured_process(args, *, shell, sig):
    # Start a process, wait a brief amount of time, then try to exit.
    proc = Popen(args, shell=shell)
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

    # Why does this hang?!!!
    run_captured_process("sleep 1d", shell=True, sig=signal.SIGINT)


if __name__ == "__main__":
    main()
