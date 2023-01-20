from contextlib import contextmanager
import inspect
from functools import partial
import multiprocessing as mp
import os
import select
import signal
import sys
from subprocess import Popen, PIPE, STDOUT
import time


def _raise_if_not_generator(worker):
    while isinstance(worker, partial):
        worker = worker.func
    if not inspect.isgeneratorfunction(worker):
        raise RuntimeError(
            f"\n\nThe following must be a generator function using the "
            f"`yield` statement: {worker}\nFor more details, see: "
            f"https://docs.python.org/3.6/howto/functional.html#generators"
            f"\n\n")


def _parallel_iwork_unordered(
        worker, values, *, process_count, pass_process_index, ctx):
    """
    Private implementation to support `parallel_work`.
    """
    # Based on: https://docs.python.org/2/library/multiprocessing.html#examples
    # "An example showing how to use queues to feed tasks to a collection of
    # worker processes and collect the results"
    assert process_count > 0

    def worker_kwargs(process_index):
        # TODO(eric.cousineau): Also pass total number of processes?
        if pass_process_index:
            return dict(process_index=process_index)
        else:
            return dict()

    inputs = ctx.Queue()
    for x in values:
        inputs.put(x)
    outputs = ctx.JoinableQueue()
    # Need a more creative token.
    stop = (('__private_stop__', None,),)

    def target(inputs, outputs, process_index):
        values_iter = iter(inputs.get, stop)
        kwargs = worker_kwargs(process_index)
        output_iter = worker(values_iter, **kwargs)
        for output in output_iter:
            outputs.put(output)
        # N.B. We must explicitly keep the process alive until all produced
        # items are consumed for libraries like torch (#5348).
        outputs.join()

    ps = []

    try:
        for i in range(process_count):
            p = ctx.Process(
                target=target,
                args=(inputs, outputs, i))
            ps.append(p)
            p.start()

        # Join, effectively flushing queue, possibly unordered.
        for p in ps:
            inputs.put(stop)

        # Poll processes and check who is done.
        while len(ps) > 0:
            # Flush out queue.
            while not outputs.empty():
                yield outputs.get()
                outputs.task_done()

            for p in ps:
                if p.exitcode is None:
                    pass
                elif p.exitcode == 0:
                    p.join()
                    ps.remove(p)
                else:
                    raise RuntimeError(
                        "Process died with code {}".format(p.exitcode))

        assert outputs.empty()
    finally:
        for p in ps:
            if p.is_alive():
                p.terminate()


class _UnitQueueIter:
    # Provides an interator which consumes and prodouces on value at a time.
    def __init__(self):
        self._next_value = None
        pass

    def put(self, value):
        assert self._next_value is None
        self._next_value = value

    def __iter__(self):
        return self

    def __next__(self):
        assert self._next_value is not None
        value = self._next_value
        self._next_value = None
        return value


# N.B. Does not work with `p.map` as the generator is not pickleable.
class _QueuedGenerator:
    # Permits converting an iterative generator (with one argument) into a
    # stateful function.
    def __init__(self, gen, **kwargs):
        self._input = _UnitQueueIter()
        self._output = gen(self._input, **kwargs)

    def __call__(self, value):
        self._input.put(value)
        return next(self._output)


def parallel_work(
        worker,
        values,
        *,
        process_count=-1,
        progress_cls=None,
        pass_process_index=False,
        async_enum=False,
        ctx=mp):
    """
    Processes iterable `values` given a generator which takes an iterable
    input and returns each output. Outputs will be returned in the same
    order of the respective inputs.

    While ``multiprocessing`` provides functions like ``mp.Pool.map``,
    ``.imap``, and ``.imap_unordered``, they do not offer easy mechanisms for
    *persistence* in each worker. (There is ``Pool(initializer=...)``, but
    requires some legwork to connect.)

    Examples of persistence:
     - When doing clutter gen, you want to pre-parse the (MultibodyPlant,
       SceneGraph) pairs, and then use them in the same process.
       (At present, these objects are not pickleable).

    @param worker
        Generator function, using `yield` keyword. See:
        https://docs.python.org/3.6/howto/functional.html#generators
    @param values
        Values for generator to operate on.
    @param process_count
        Number of CPUs; if -1 or None, use all available; if 0, use this
        process (useful for debugging / simplicity) do not use multiprocessing.
        Note that process_count=1 is distinct from process_count=0, in that it
        will spin up a separate process and require transfer via pickling.
    @param progress_cls
        Wraps the iterator in the form of `progress_cls(it, total=len(values)`
        (e.g. ``tqdm.tqdm`` or ``tqdm.tqdm_notebook``).
    @param pass_process_index
        Pass index of given process (0...process_count-1) as
        ``worker(..., process_index=x)``. This can be used to help delegate
        work to certain GPUs.
    @param async_enum
        If True, will pass back an iterator, which will yield (index, output)
        rather than just output.
    @param ctx
        The multiprocessing context to use. Default is mp (multiprocessing)
        itself, but it can also be:
        - mp.dummy - to use threading instead
        - mp.get_context(method) to use a different start method (e.g. "fork")
        - torch.multiprocessing to use torch's lightweight customizations
        (though for Python 3.6+, torch registers its type-specific hooks in mp
        itself).

    @warning At present, since `parallel_work` uses local nested functions with
        closures (which cannot be pickled), the "spawn" method cannot yet be
        used.

    Example:

        import multiprocessing as mp
        import random
        import time

        def worker(values):
            # Count how much work each worker does.
            count = 0
            for value in values:
                time.sleep(random.uniform(0.05, 0.1))
                count += 1
                yield (mp.current_process().name, count, value)

        outputs = parallel_work(worker, range(5), process_count=3))

    Sample contents of `outputs`, with `process_count=3`:

        [('Process-1', 1, 0),
         ('Process-2', 1, 1),
         ('Process-3', 1, 2),
         ('Process-2', 2, 3),
         ('Process-1', 2, 4)]

    With `process_count=0`:

         [('MainProcess', 1, 0),
         ('MainProcess', 2, 1),
         ('MainProcess', 3, 2),
         ('MainProcess', 4, 3),
         ('MainProcess', 5, 4)]
    """
    # TODO(eric.cousineau): Get rid of `process_count=None`, and only allow
    # `=-1` to use all CPUs.
    # TODO(eric.cousineau): Use pickle-compatible wrapper functions.

    _raise_if_not_generator(worker)
    n = len(values)
    if not hasattr(values, "__getitem__"):
        raise RuntimeError(f"`values` must be indexable: {type(values)}")
    if process_count is None or process_count == -1:
        process_count = os.cpu_count()

    def wrap(it):
        if progress_cls:
            return progress_cls(it, total=n)
        else:
            return it

    if process_count != 0:

        def worker_wrap(indices, **kwargs):
            # Preserve iteration using queued generator so that we can monitor
            # its progression using the `progress_cls` wrapper.
            worker_queued = _QueuedGenerator(worker, **kwargs)
            for i in indices:
                # TODO(eric.cousineau): It seems very inefficient to catpure
                # `values` here (needs to be pickled in capture).
                # *However*, if I change the inputs to
                #   pairs = list(enumerate(values))
                # then `process_util_torch_test` freezes...
                output = worker_queued(values[i])
                yield (i, output)

        indices = range(len(values))
        enum_outputs_iter = _parallel_iwork_unordered(
            worker_wrap,
            indices,
            process_count=process_count,
            pass_process_index=pass_process_index,
            ctx=ctx,
        )
        enum_outputs_iter = wrap(enum_outputs_iter)

        if async_enum:
            return enum_outputs_iter

        enum_outputs = list(enum_outputs_iter)
        assert len(enum_outputs) == n, len(enum_outputs)
        # Re-order.
        outputs = n * [None]
        for i, output in enum_outputs:
            outputs[i] = output
        return outputs
    else:
        kwargs = dict()
        if pass_process_index:
            kwargs = dict(process_index=0)
        outputs_iter = wrap(worker(values, **kwargs))

        if async_enum:
            return enumerate(outputs_iter)

        return list(outputs_iter)


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
        register_lazy_shutdown(del_f)

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
            cur = os.read(f.fileno(), chunk_size)
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
            **kwargs):
        # Python processes don't like buffering by default.
        args = ["env", "PYTHONUNBUFFERED=1", "stdbuf", "-o0"] + args
        self._args = args
        if simple_encoding:
            kwargs.update(encoding="utf8", universal_newlines=True, bufsize=1)
        proc = Popen(args, stdin=PIPE, stdout=PIPE, stderr=stderr, **kwargs)
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

    def add(self, name, *args, on_new_text="default", **kwargs):
        """Adds a process `name`."""
        assert name not in self._proc_map, name
        prefix = "[{}] ".format(name)
        if on_new_text == "default":
            on_new_text = partial(print_prefixed, prefix=prefix)
        proc = CapturedProcess(*args, on_new_text=on_new_text, **kwargs)
        self._proc_map[name] = proc
        return proc

    def get(self, name):
        """Gets a specified process."""
        return self._proc_map[name]

    def remove(self, name, close=True):
        """
        Removes a process.
        @param close If true, will terminate the process.
        @return The process that was removed.
        """
        proc = self.get(name)
        if close:
            signal_processes([proc])
        del self._proc_map[name]
        return proc

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

    def wait_for_any_exit(self):
        while self.poll() == {}:
            time.sleep(0.05)
        return self.poll()


_shutdowns_stack = []


def register_lazy_shutdown(f):
    """
    Inelegant way to try and ensure resources are cleaned up when not created
    within an explicit context manager.

    Example: When allocating a resource towards an object, which may try to
    rely on __del__. However, this hook may not always be executed as desired
    when an interpreter exits:
    https://docs.python.org/3.6/reference/datamodel.html#object.__del__
    """
    # TODO(eric): See if we can be more elegant about this.
    # TODO(eric): Use atexit?!
    if len(_shutdowns_stack) > 0:
        # TODO(eric): Warn or error about this.
        _shutdowns_stack[-1].append(f)


@contextmanager
def lazy_shutdown_context():
    """
    Use this to explicitly clean up anything registered with
    `register_lazy_shutdown()`. Generally, you should place this at the
    top-level.
    See additional description in `register_lazy_shutdown()`.

    Example:

        if __name__ == "__main__":
            with lazy_shutdown_context():
                main()
    """
    cur_shutdowns = []
    _shutdowns_stack.append(cur_shutdowns)
    expected_len = len(_shutdowns_stack)
    yield
    assert len(_shutdowns_stack) == expected_len
    assert _shutdowns_stack[-1] is cur_shutdowns
    del _shutdowns_stack[-1]
    for f in cur_shutdowns:
        f()
