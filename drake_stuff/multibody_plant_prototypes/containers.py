"""Provides simple extension containers."""

from contextlib import contextmanager
import itertools

import numpy as np


class AttrDict(dict):
    """Access / mutate dictionary entries as attributes."""
    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value

    @staticmethod
    def create_recursive(other):
        assert isinstance(other, dict), type(other)
        out = AttrDict()
        for k, v in other.items():
            if isinstance(v, dict):
                v = AttrDict.create_recursive(v)
            out[k] = v
        return out

    def asdict(self):
        out = dict()
        for k, v in self.items():
            if isinstance(v, dict) and type(v) != dict:
                v = dict(v)
            out[k] = v
        return out


class SortedSet(set):
    """A set that maintains strict sorting when iterated upon."""
    def __init__(self, *args, sorted=sorted, **kwargs):
        # TODO(eric): Add `strict` option to prevent duplicates.
        self._sorted = sorted
        super().__init__(*args, **kwargs)

    def __iter__(self):
        original = super().__iter__()
        return iter(self._sorted(original))


class SortedDict(dict):
    """A dict that maintains strict sorting when iterated upon.
    All values are returned according to the sorting of keys, not the
    values."""
    def __init__(self, *args, sorted_keys=sorted, **kwargs):
        # TODO(eric): Add `strict` option to prevent duplicates.
        self._sorted_keys = sorted_keys
        super().__init__(*args, **kwargs)

    def __iter__(self):
        original = super().__iter__()
        return iter(self._sorted_keys(original))

    def __repr__(self):
        items_str = ", ".join(f"{repr(k)}: {repr(v)}" for k, v in self.items())
        return f"SortedDict({{{items_str}}})"

    def items(self):
        out = []
        for key in self:
            out.append((key, self[key]))
        return out

    def keys(self):
        return list(iter(self))

    def values(self):
        items = self.items()
        return [value for (_, value) in items]


def dict_items_zip(*items):
    """
    Provides `zip()`-like functionality for the items of a list of
    dictionaries. This requires that all dictionaries have the same keys
    (though possibly in a different order).

    Returns:
        Iterable[key, values], where ``values`` is a tuple of the value from
        each dictionary.
    """
    if len(items) == 0:
        # Return an empty iterator.
        return
    first = items[0]
    assert isinstance(first, dict)
    check_keys = set(first.keys())
    for item in items[1:]:
        assert isinstance(item, dict)
        assert set(item.keys()) == check_keys
    for k in first.keys():
        values = tuple(item[k] for item in items)
        yield k, values


def strict_zip(a, b):
    # Ensures that both containers have the same length. Normal `zip()`
    # functionality will stop when it reached the end of the shortest
    # container.
    assert len(a) == len(b)
    return zip(a, b)


def take_first(iterable):
    """
    Robustly gets the first item from an iterable and returns it.
    You should always use this isntead of `next(iter(...))`; e.g. instead of

        my_first = next(iter(container))

    you should instead do:

        my_first = take_first(container)
    """
    first, = itertools.islice(iterable, 1)
    return first


@contextmanager
def printoptions(**kwargs):
    # Remove once we have numpy>=1.15.
    old = np.get_printoptions()
    np.set_printoptions(**kwargs)
    yield
    np.set_printoptions(**old)
