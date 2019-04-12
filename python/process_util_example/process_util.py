from functools import partial
import multiprocessing as mp
from os import read
import select
import signal
import sys
from subprocess import check_call, Popen, PIPE, STDOUT


def parallel_iwork_unordered(worker, values, process_count=None):
    """Processes a set of values given a generator which takes an iterative
    input and returns each output. Values will be unordered.

    @param worker
        Generator function, using `yield` keyword.
    @param values
        Values for generator to operate on.
    @param process_count
        Number of CPUs; if None, use all available; if 0, do not use
        multiprocessing (which is useful for debugging).

    Example:

        from multiprocessing import current_process
        import random
        import time

        def worker(values):
            # Count how much work each worker does.
            count = 0
            for value in values:
                time.sleep(random.uniform(0.05, 0.1))
                count += 1
                yield (current_process().name, count, value)

        outputs = list(parallel_iwork_unordered(
            worker, range(5), process_count=3))

    Sample contents of `outputs`:

        [('Process-2', 1, 1),
         ('Process-1', 1, 0),
         ('Process-3', 1, 2),
         ('Process-2', 2, 3),
         ('Process-1', 2, 4)]

    If setting `process_count=0`:

         [('MainProcess', 1, 0),
         ('MainProcess', 2, 1),
         ('MainProcess', 3, 2),
         ('MainProcess', 4, 3),
         ('MainProcess', 5, 4)]
    """
    # Based on: https://docs.python.org/2/library/multiprocessing.html#examples
    # "An example showing how to use queues to feed tasks to a collection of
    # worker processes and collect the results"
    if process_count == 0:
        # Simple case.
        for output in worker(values):
            yield output
        raise StopIteration

    inputs = mp.Queue()
    map(inputs.put, values)
    outputs = mp.Queue()
    # Need a more creative token.
    stop = (('__private_stop__', None,),)

    def target(inputs, outputs):
        values_iter = iter(inputs.get, stop)
        output_iter = worker(values_iter)
        for output in output_iter:
            outputs.put(output)

    if process_count is None:
        process_count = mp.cpu_count()
    ps = []

    try:
        for i in xrange(process_count):
            p = mp.Process(
                target=target,
                args=(inputs, outputs))
            ps.append(p)
            p.start()

        # Join, effectively flushing queue, possibly unordered.
        for p in ps:
            inputs.put(stop)

        # Poll processes and check who is done.
        while len(ps) > 0:
            for p in ps:
                if p.exitcode is None:
                    pass
                elif p.exitcode == 0:
                    p.join()
                    ps.remove(p)
                else:
                    raise RuntimeError(
                        "Process died with code {}".format(p.exitcode))
            # Flush out queue.
            while not outputs.empty():
                yield outputs.get()

        # Flush again.
        while not outputs.empty():
            yield outputs.get()
    finally:
        for p in ps:
            if p.is_alive():
                p.terminate()


class _UnitQueueIter(object):
    # Provides an interator which consumes and prodouces on value at a time.
    def __init__(self):
        self._next_value = None
        pass

    def put(self, value):
        assert self._next_value is None
        self._next_value = value

    def __iter__(self):
        return self

    def next(self):
        assert self._next_value is not None
        value = self._next_value
        self._next_value = None
        return value


# N.B. Does not work with `p.map` as the generator is not pickleable.
class _QueuedGenerator(object):
    # Permits converting an iterative generator (with one argument) into a
    # stateful function.
    def __init__(self, gen):
        self._input = _UnitQueueIter()
        self._output = gen(self._input)

    def __call__(self, value):
        self._input.put(value)
        return self._output.next()


def parallel_work(worker, values, progress_cls=None, process_count=None):
    """
    Wraps `parallel_generator`, but outputs are ordered.

    @param progress_cls
        Wraps the iterator in the form of `progress_cls(it, total=len(values)`
        (e.g. `tqdm` or `tqdm_notebook`).

    This differs from `multiprocessing.map` in that it permits persistence
    by a direct generator, rather than relying on an `init` function.

    Example:

        from multiprocessing import current_process
        import random
        import time

        def worker(values):
            # Count how much work each worker does.
            count = 0
            for value in values:
                time.sleep(random.uniform(0.05, 0.1))
                count += 1
                yield (current_process().name, count, value)

        outputs = parallel_work(
            worker, ["hey", "world", "look", "at", "me"], process_count=3)

    Sample contents of `outputs`:

        [('Process-1', 1, 'hey'),
         ('Process-2', 1, 'world'),
         ('Process-3', 1, 'look'),
         ('Process-3', 2, 'at'),
         ('Process-2', 2, 'me')]
    """
    n = len(values)

    def wrap(it):
        if progress_cls:
            return progress_cls(it, total=n)
        else:
            return it

    if process_count != 0:

        def worker_wrap(indices):
            # Preserve iteration using queued generator so that we can monitor
            # its progression using the `progress_cls` wrapper.
            worker_queued = _QueuedGenerator(worker)
            for i in indices:
                output = worker_queued(values[i])
                yield (i, output)

        indices = range(len(values))
        enum_outputs_iter = parallel_iwork_unordered(
            worker_wrap, indices, process_count)
        enum_outputs = list(wrap(enum_outputs_iter))
        # Re-order.
        outputs = n * [None]
        for i, output in enum_outputs:
            outputs[i] = output
        return outputs
    else:
        return list(wrap(worker(values)))


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


class on_context_exit(object):
    """
    Calls a function with given arguments when a context is left, or when
    this object is garbage collected.
    Consider using `contextmanager.closing` for stuiff that has `.close()`.

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
    def __init__(self, f):
        self._on_exit = f

    def __enter__(self):
        pass

    def __exit__(self, *args):
        if self._on_exit:
            self._on_exit()
            self._on_exit = None

    def __del__(self):
        if self._on_exit:
            self._on_exit()
            self._on_exit = None


def read_available(f, timeout=0.0, chunk_size=1024, empty=""):
    """
    Reads all available data on a given file. Useful for using PIPE with Popen.

    @param timeout Timeout for `select`.
    @param chunk_size How much to try and read.
    @param empty Starting point / empty value. Default value is "" (good for
    strings).
    """
    readable, _, _ = select.select([f], [], [f], timeout)
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


class StreamCollector(object):
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
                text_new += read_available(stream, timeout=timeout)
            if text_new:
                for on_new_text in self._on_new_text:
                    on_new_text(text_new)
            self._text += text_new
        return self._text


class CapturedProcess(object):
    """
    Captures a process's stdout and stderr to a StreamCollector and provides
    (non-blocking) polling access to the latest output.
    Useful for interacting with input / output in a simple fashion.

    Note: This has only been designed for simple offline scripts, not for
    controlling realtime stuff.
    For complex state machine process interaction, use `pexpect`.
    """
    def __init__(
            self, args, stderr=STDOUT, on_new_text=None, **kwargs):
        # Python processes don't like buffering by default.
        args = ["stdbuf", "--output=0"] + args
        self._args = args
        proc = Popen(args, stdin=PIPE, stdout=PIPE, stderr=stderr, **kwargs)
        self.proc = proc
        # N.B. Do not pass `self` inside the lambda to simplify references for
        # garbage collection.
        self.scope = on_context_exit(
            lambda: signal_processes([proc]))
        streams = [proc.stdout]
        if proc.stderr:  # `None` if `stderr=STDOUT` is specified via `Popen`
            streams += [proc.stderr]
        self.output = StreamCollector(streams, on_new_text=on_new_text)

    def __repr__(self):
        return "<CapturedProcess {}>".format(self._args)

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
        this is a no-op."""
        with self.scope:
            pass


class CapturedProcessGroup(object):
    """Groups CapturedProcess objects, and allows for simpler usage."""
    def __init__(self):
        self._proc_map = dict()

    def add(self, name, *args, **kwargs):
        """Adds a process `name`."""
        assert name not in self._proc_map, name
        prefix = "[{}] ".format(name)
        on_new_text = kwargs.pop("on_new_text", "default")
        if on_new_text == "default":
            on_new_text = partial(print_prefixed, prefix=prefix)
        proc = CapturedProcess(*args, on_new_text=on_new_text, **kwargs)
        self._proc_map[name] = proc
        return proc

    def get(self, name):
        """Gets a specified process."""
        return self._proc_map[name]

    def poll(self):
        """Polls all processes. Any processes that have stopped running are
        returned within a map, {name: status}."""
        stats = {}
        for name, proc in self._proc_map.items():
            stat = proc.poll()
            if stat is not None:
                stats[name] = stat
        return stats

    def close(self):
        """Closes all processes."""
        signal_processes([x.proc for x in self._proc_map.values()])
