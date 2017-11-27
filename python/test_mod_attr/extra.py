#!/usr/bin/env python

import sys

def check_direct():
    import mod
    print(mod.extra)

def check_from():
    # Prints true, that it is 
    from mod import extra

check_direct()
del sys.modules["mod"]
check_from()

# print(hasattr(mod, "extra"))
# print(dir(mod))
