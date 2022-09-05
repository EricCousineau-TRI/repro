# -*- python -*-
# vi: set ft=python :

"""
Simple containr utilities.
"""

# TODO(eric.cousineau): Use bazel_skylib at some point.

def sort_friendly(x):
    # Sorts. If x is a depset, will turn it into a list.
    if type(x) == "depset":
        x = x.to_list()
    return sorted(x)

def combine_and_sort(depsets):
    """Combine and sort depsets. Useful if you want to check length, etc."""
    return sort_friendly(depset(transitive = depsets))

def uniq(xs):
    return sort_friendly(depset(xs))

def set_diff(a, b):
    # Returns a - b, that is, elements in a that are not in b.
    a_not_in_b = []
    a = sort_friendly(a)
    b = sort_friendly(b)
    for a_i in a:
        if a_i not in b:
            a_not_in_b.append(a_i)
    return a_not_in_b

def to_str_list(xs, truncate_at):
    if len(xs) > truncate_at:
        remaining = len(xs) - truncate_at
        xs = xs[:truncate_at] + ["<truncated; {} remaining>".format(remaining)]
    return [str(x) for x in xs]

def indent(s, prefix):
    return "\n".join([prefix + line for line in s.splitlines()])
