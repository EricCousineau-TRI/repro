"""
Utilities that should be synchronized with:
https://drake.mit.edu/python_bindings.html#debugging-with-the-python-bindings
"""

import bdb
from contextlib import contextmanager
import pdb
import sys
import traceback


@contextmanager
def launch_pdb_on_exception():
    """
    Provides a context that will launch interactive pdb console automatically
    if an exception is raised.

    Example usage with @iex decorator below:

        @iex
        def my_bad_function():
            x = 1
            assert False

        my_bad_function()
        # Should bring up debugger at `assert` statement.
    """
    # Adapted from:
    # https://github.com/gotcha/ipdb/blob/fc83b4f5f/ipdb/__main__.py#L219-L232

    # TODO(eric.cousineau): If stdin is not a tty, should bail faster?

    try:
        yield
    except bdb.BdbQuit:
        pass
    except (Exception, SystemExit):
        traceback.print_exc()
        _, _, tb = sys.exc_info()
        pdb.post_mortem(tb)
        # Resume original execution.
        raise


# See docs for `launch_pdb_on_exception()`.
iex = launch_pdb_on_exception()
