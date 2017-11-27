#!/usr/bin/env python

import mod

print(mod.value)
print(mod.extra)
print(mod.extra)

before = None
before = set(locals().keys())

from mod import *
new = set(locals().keys()) - before
print("\n".join(sorted(new)))
