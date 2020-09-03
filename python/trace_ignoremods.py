import collections
import copy
import sys
import trace


def make_tracer():
    ignoredirs = ("/home")
    only_mods = {"copy"}
    ignoremods = list(sorted(set(sys.modules.keys()) - only_mods))

    tracer = trace.Trace(trace=1, count=0, ignoredirs=ignoredirs, ignoremods=ignoremods)
    return tracer


def main():
    # Traced.
    copy.deepcopy("hello")
    # Not traced.
    collections.OrderedDict()


assert __name__ == "__main__"
tracer = make_tracer()
tracer.runfunc(main)
