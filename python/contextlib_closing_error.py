import contextlib



def closing_multiple(*things):
    """
    Plural version of `contextlib.closing(thing)`.

    If any exceptions occur when closing, the last exception will be re-thrown.
    """
    exit_stack = contextlib.ExitStack()
    for thing in things:
        closing_context = contextlib.closing(thing)
        exit_stack.enter_context(closing_context)
    return exit_stack


@contextlib.contextmanager
def closing_multiple_custom(*things):
    # Equivalent to above, it seems.
    try:
        yield
    finally:
        e = None
        for thing in things:
            try:
                thing.close()
            except Exception as new_e:
                e = new_e
        if e is not None:
            raise e


class GoodClose:
    def __init__(self):
        self.was_closed = False

    def close(self):
        self.was_closed = True


class ThrowsOnClose:
    counter = 0

    def close(self):
        ThrowsOnClose.counter += 1
        assert False, f"counter: {ThrowsOnClose.counter}"


def main():
    good = GoodClose()
    # See if `good` gets closed, even if `bad` has an error.
    try:
        with closing_multiple_custom(ThrowsOnClose(), ThrowsOnClose(), good):
            pass
    finally:
        print(f"was_closed: {good.was_closed}")


if __name__ == "__main__":
    main()


"""
$ python contextlib_closing_error.py 2>&1 | sed 's#'${PWD}'#${PWD}#g'
was_closed: True
Traceback (most recent call last):
  File "${PWD}/contextlib_closing_error.py", line 61, in <module>
    main()
  File "${PWD}/contextlib_closing_error.py", line 54, in main
    with closing_multiple_custom(ThrowsOnClose(), ThrowsOnClose(), good):
  File "/usr/lib/python3.10/contextlib.py", line 142, in __exit__
    next(self.gen)
  File "${PWD}/contextlib_closing_error.py", line 31, in closing_multiple_custom
    raise e
  File "${PWD}/contextlib_closing_error.py", line 27, in closing_multiple_custom
    thing.close()
  File "${PWD}/contextlib_closing_error.py", line 47, in close
    assert False, f"counter: {ThrowsOnClose.counter}"
AssertionError: counter: 2
"""
