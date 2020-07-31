from __future__ import absolute_import, print_function

# Define these first, as they are used in `cpp_types.cc`.

import ctypes
import numpy as np


def _get_type_name(t):
    # Gets scoped type name as a string.
    prefix = t.__module__ + "."
    if prefix == "__builtin__.":
        prefix = ""
    return prefix + t.__name__


class _StrictMap(object):
    def __init__(self):
        self._values = dict()

    def _strict_key(self, key):
        # Ensures keys are strictly scoped to the values (for literals).
        return (type(key), key)

    def add(self, key, value):
        skey = self._strict_key(key)
        assert skey not in self._values, "Already added: {}".format(skey)
        self._values[skey] = value

    def get(self, key, default):
        skey = self._strict_key(key)
        return self._values.get(skey, default)


# Load and import type registry.
from ._cpp_types import _type_registry

def type_canonical(t):
    return _type_registry.GetPyTypeCanonical(t)

def type_name(t):
    return _type_registry.GetName(t)

def types_canonical(param):
    """Gets canonical parameter pack that makes it simple to mesh with
    C++ types. """
    return tuple(map(type_canonical, param))

def type_names(param):
    return tuple(map(type_name, param))
