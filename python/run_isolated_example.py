import functools
import multiprocessing as mp


def _run_isolated_target(queue, func, *args, **kwargs):
    queue.put(func(*args, **kwargs))


def run_isolated(ctx, func, *args, **kwargs):
    # Use multiprocessing to spawn a new process to check things like pickling
    # in a separate process (where certain modules may not be imported).
    queue = ctx.Queue(maxsize=1)
    target = functools.partial(
        _run_isolated_target, queue, func, *args, **kwargs
    )
    proc = ctx.Process(target=target)
    proc.start()
    proc.join()
    if proc.exitcode != 0:
        raise RuntimeError(f"Process died with code {proc.exitcode}")
    return queue.get()


...

import unittest

class Test(unittest.TestCase):
    @staticmethod
    def check_across_proc_boundary():
        # cannot use `self`
        import some_crazy_module
        assert something_didnt_fail()
        return some_pickle_friendly_value

    def test_across_proc_boundary(self):
        some_pickle_friendly_value = run_isolated(
            mp.get_context("spawn"), 
            self.check_across_proc_boundary,
        )
        self.assertSomething(some_pickle_friendly_value)
