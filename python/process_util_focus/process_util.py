from functools import partial
import os
from os import read
import select
import shlex
import signal
from subprocess import PIPE, Popen, STDOUT
import sys
import time


def signal_processes(
        process_list, sig=signal.SIGINT, block=True, close_streams=True):
    """
    Robustly sends a singal to processes that are still alive. Ignores status
    codes.

    @param process_list List[Popen] Processes to ensure are sent a signal.
    @param sig Signal to send. Default is `SIGINT`.
    @param block Block until process exits.
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


class on_context_exit:
    """
    Calls a function with given arguments when a context is left, or when
    this object is garbage collected. `del_f` can be used to use a separate
    routine if garbage-collected.

    Example:

        from subprocess import Popen
        import time
        p = Popen(["sleep", "10"])
        with on_context_exit(lambda: signal_processes([p])):
            # Do things.
            time.sleep(1.)
        # When exiting (due to success or exception failure), the process will
        # be killed via SIGINT.
    """
    def __init__(self, f, del_f="same"):
        self._done = False
        self._on_exit = f
        if del_f == "same":
            del_f = f
        self._on_del = del_f

    def _check_and_mark_done(self):
        if self._done:
            return False
        else:
            self._done = True
            return True

    def __enter__(self):
        pass

    def __exit__(self, *args):
        if self._on_exit and self._check_and_mark_done():
            self._on_exit()

    def __del__(self):
        if self._on_del and self._check_and_mark_done():
            self._on_del()


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
            cur = read(f.fileno(), chunk_size)
            out += cur
            if len(cur) < chunk_size:
                break
    return out


def print_prefixed(text, prefix="", streams=[sys.stdout]):
    """
    Prints non-empty text to a list of streams with a prefix.
    """
    if text.endswith("\n"):
        text = text[:-1]
    if not streams or not text:
        return
    prefixed = "\n".join([prefix + line for line in text.split("\n")])
    for stream in streams:
        stream.write(prefixed + "\n")
        stream.flush()


class StreamCollector:
    """
    Collects available text from a stream (e.g. a PIPE from Popen).

    @param on_new_text When new text is received; can use
    `print_prefixed`.  If this is iterable, each item will
    be called with the newly received text.

    @param on_new_text Callback functor(s) to be invoked when new text
    is received. Can be a scalar (for a single functor) or an iterable
    (for multiple functors, all of which will be called for all new
    text). See `print_prefixed` above for an example functor
    implementation.
    """

    def __init__(self, streams, on_new_text=tuple()):
        self._streams = streams
        self._text = ""
        try:
            # Check if on_new_text is iterable
            iter(on_new_text)
            self._on_new_text = on_new_text
        except TypeError:
            self._on_new_text = (on_new_text,)

    def clear(self):
        self._text = ""

    def get_text(self, timeout=0.):
        """
        Gets current text.
        @param timeout Timeout for polling each stream. If None, will not
        update.
        """
        if timeout is not None:
            # Get available bytes from streams.
            text_new = ""
            for stream in self._streams:
                if stream.closed:
                    continue
                data = read_available(stream, timeout=timeout)
                if isinstance(data, bytes):
                    data = data.decode("utf8")
                text_new += data
            if text_new:
                for on_new_text in self._on_new_text:
                    if on_new_text is None:
                        continue
                    on_new_text(text_new)
            self._text += text_new
        return self._text


class CapturedProcess:
    """
    Captures a process's stdout and stderr to a StreamCollector and provides
    (non-blocking) polling access to the latest output.
    Useful for interacting with input / output in a simple fashion.

    Note: This has only been designed for simple offline scripts, not for
    controlling realtime stuff.
    For complex state machine process interaction, use `pexpect`.

    @param on_new_text
        See `StreamCollector` doc. This is also called on `poll()`.
    @param simple_encoding
        Adds the following to `kwargs` (useful for text-based programs):
            dict(encoding="utf8", universal_newlines=True, bufsize=1)
    @param **kwargs Additional arguments to `Popen` constructor.
    """
    def __init__(
            self, args, stderr=STDOUT, on_new_text=None, simple_encoding=True,
            verbose=False, shell=False, env=None, **kwargs):
        # Python processes don't like buffering by default.
        if verbose:
            if shell:
                cmd = args
            else:
                cmd = shlex.join(args)
            print(f"+ {cmd}", file=sys.stderr)

        if env is None:
            env = dict(os.environ)
        else:
            env = dict(env)
        if "PYTHONUNBUFFERED" not in env:
            env["PYTHONUNBUFFERED"] = "1"

        self._args = args
        if simple_encoding:
            kwargs.update(encoding="utf8", universal_newlines=True, bufsize=1)
        proc = Popen(
            args, stdin=PIPE, stdout=PIPE, stderr=stderr, shell=shell, env=env, **kwargs
        )
        self.proc = proc
        # N.B. Do not pass `self` inside the lambda to simplify references for
        # garbage collection.
        # TODO(eric.cousineau): this still doesn't work... how to do RAII w/o
        # needing a shitton of `with` statements? process groups?
        self.scope = on_context_exit(
            lambda: signal_processes([proc]),
            # Have a more assured exit, in case SIGINT does not make the
            # process die quickly enough (or at all, in the case of ROS).
            del_f=lambda: signal_processes([proc], signal.SIGABRT))
        streams = [proc.stdout]
        if proc.stderr:  # `None` if `stderr=STDOUT` is specified via `Popen`
            streams += [proc.stderr]
        self.output = StreamCollector(streams, on_new_text=on_new_text)

    def __repr__(self):
        return "<CapturedProcess {}>".format(self._args)

    def pid(self):
        return self.proc.pid

    def poll(self):
        """Polls process, returning exit code."""
        self.output.get_text()  # Flush text for debugging.
        return self.proc.poll()

    def wait(self):
        """Waits until a process ends, polling the process (ensuring that
        `on_new_text` is called). Returns exit status."""
        while self.poll() is None:
            pass
        return self.poll()

    def close(self):
        """Attempts to signal process to close. If process is already closed,
        this is a no-op. Returns exit status."""
        with self.scope:
            pass
        return self.poll()


class CapturedProcessGroup:
    """Groups CapturedProcess objects, and allows for simpler usage."""
    def __init__(self):
        self._proc_map = dict()

    def add(self, name, args, on_new_text="default", verbose=False, **kwargs):
        """Adds a process `name`."""
        assert name not in self._proc_map, name
        prefix = "[{}] ".format(name)
        if on_new_text == "default":
            on_new_text = partial(print_prefixed, prefix=prefix)
        proc = CapturedProcess(
            args, on_new_text=on_new_text, verbose=verbose, **kwargs
        )
        self._proc_map[name] = proc
        return proc

    def has(self, name):
        return name in self._proc_map

    def get(self, name):
        """Gets a specified process."""
        return self._proc_map[name]

    def remove(self, name, close=True):
        """
        Removes a process.
        @param close If true, will terminate the process.
        @return The process that was removed.
        """
        wrapped = self.get(name)
        if close:
            signal_processes([wrapped.proc])
        del self._proc_map[name]
        return wrapped

    def poll(self):
        """Polls all processes. Any processes that have stopped running are
        returned within a map, {name: status}."""
        stats = {}
        for name, proc in self._proc_map.items():
            stat = proc.poll()
            if stat is not None:
                stats[name] = stat
        return stats

    def close(self, *, sig=signal.SIGINT):
        """Closes all processes."""
        signal_processes([x.proc for x in self._proc_map.values()], sig=sig)

    def wait_for_any_exit(self):
        while self.poll() == {}:
            time.sleep(0.05)
        return self.poll()
