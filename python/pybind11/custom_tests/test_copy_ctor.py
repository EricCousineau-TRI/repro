import copy_ctor as m

def main():
    c = m.Custom(1)
    c2 = m.Custom(c)
    print(int(c))

import trace, sys
sys.stdout = sys.stderr
tracer = trace.Trace(ignoredirs=sys.path, trace=1, count=0)
# N.B. `run` seems to use the wrong locals / globals...
tracer.runfunc(main)
