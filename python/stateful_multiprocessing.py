"""
Abstract example; ultimately, to be applied for using multiprocessing with a
CPython C extension (e.g. pydrake).
"""

from collections import namedtuple
import functools
import multiprocessing as mp
import os

# N.B. `functools.cache` is in Python 3.9.
# Instead, we its functional equivalent, `functools.lru_cache

ExampleResult = namedtuple("ExampleResult", ["pid", "counter", "value"])


class State:
    def __init__(self):
        self.counter = 0


@functools.lru_cache(maxsize=None)
def per_process_global_state():
    # Setup, effectively globally.
    # Do NOT call this until you've started the pool.
    return State()


def per_item_step(value):
    state = per_process_global_state()
    state.counter += 1
    return ExampleResult(os.getpid(), state.counter, value)


def multi_step(pool, values):
    return pool.map(per_item_step, values)


def main():
    # N.B. You would want to store the `pool` somewhere.
    pool = mp.Pool(3, initializer=per_process_global_state)
    with pool:
        for i in range(1, 5):
            values = [100 * i + k for k in range(6)]
            outputs = multi_step(pool, values)
            print("\n".join(map(str, outputs)))
            print()


assert __name__ == "__main__"
main()


"""
Example output; for this example, it's not important what the value of counter
is, just that you have persistence across workers.

ExampleResult(pid=23812, counter=1, value=100)
ExampleResult(pid=23813, counter=1, value=101)
ExampleResult(pid=23814, counter=1, value=102)
ExampleResult(pid=23812, counter=2, value=103)
ExampleResult(pid=23813, counter=2, value=104)
ExampleResult(pid=23814, counter=2, value=105)

ExampleResult(pid=23812, counter=3, value=200)
ExampleResult(pid=23813, counter=3, value=201)
ExampleResult(pid=23814, counter=3, value=202)
ExampleResult(pid=23812, counter=4, value=203)
ExampleResult(pid=23813, counter=4, value=204)
ExampleResult(pid=23814, counter=4, value=205)

ExampleResult(pid=23812, counter=5, value=300)
ExampleResult(pid=23813, counter=5, value=301)
ExampleResult(pid=23814, counter=5, value=302)
ExampleResult(pid=23812, counter=6, value=303)
ExampleResult(pid=23813, counter=6, value=304)
ExampleResult(pid=23814, counter=6, value=305)

ExampleResult(pid=23812, counter=7, value=400)
ExampleResult(pid=23813, counter=7, value=401)
ExampleResult(pid=23814, counter=7, value=402)
ExampleResult(pid=23812, counter=8, value=403)
ExampleResult(pid=23813, counter=8, value=404)
ExampleResult(pid=23814, counter=8, value=405)
"""
