#!/usr/bin/env python

import sys

def check_direct():
    import mod
    print(mod.extra)

def check_from():
    # Prints true, that it is 
    from mod import extra
    print(extra)

check_direct()
del sys.modules["mod"]
check_from()
del sys.modules["mod"]
check_from()
del sys.modules["mod"]

# Check star.
from mod import *
