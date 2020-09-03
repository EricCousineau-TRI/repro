import collections
import copy
import sys
import trace


def make_tracer(only_mods):
    ignoredirs = ()
    all_mods = set(sys.modules.keys())
    ignoremods = set()
    for mod in all_mods:
        good = False
        for only_mod in only_mods:
            if mod == only_mod or mod.startswith(f"{only_mod}.") or only_mod.startswith(f"{mod}."):
                good = True
                break
        if not good:
            ignoremods.add(mod)
    print("\n".join(sorted(ignoremods)))

    tracer = trace.Trace(trace=1, count=0, ignoredirs=ignoredirs, ignoremods=ignoremods)
    return tracer


def main():
    # Not traced.
    copy.deepcopy("hello")
    # Still traced?
    collections.namedtuple("A", ())


assert __name__ == "__main__"
tracer = make_tracer(only_mods={"copy"})
tracer.runfunc(main)
