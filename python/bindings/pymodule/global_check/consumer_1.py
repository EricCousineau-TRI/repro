from __future__ import absolute_import, division, print_function
import sys

# From DLFCN.py
# Included from bits/dlfcn.h
RTLD_LAZY = 0x00001
RTLD_NOW = 0x00002
RTLD_BINDING_MASK = 0x3
RTLD_NOLOAD = 0x00004
RTLD_GLOBAL = 0x00100
RTLD_LOCAL = 0
RTLD_NODELETE = 0x01000

sys.setdlopenflags(RTLD_NOW | RTLD_GLOBAL)
from ._consumer_1 import *

# # import imp
# # _tmp = imp.load_dynamic("_consumer_1", "_consumer_1.so")
# print(__name__)
# pieces = __name__.split('.')
# print(pieces)
# c_name = '.'.join(pieces[:-1]) + "._consumer_1"
# print(c_name)

# cur = __import__(c_name)
# # print(mod.__file__)
# print(mod)

# # print(mod.__dict__.keys())
# # locals().update(mod.__dict__)
# # do_stuff_1 = mod.do_stuff_1
# # from ._consumer_1 import *
