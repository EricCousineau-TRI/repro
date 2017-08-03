#!/usr/bin/env python
import sys
import ctypes

# Will this work on Mac?

# @ref https://stackoverflow.com/a/19664308

class ShareSymbols(object):
    def __init__(self):
        self.old_flags = sys.getdlopenflags()
        self.new_flags = self.old_flags | ctypes.RTLD_GLOBAL

    def __enter__(self, *args, **kwargs):
        sys.setdlopenflags(self.new_flags)

    def __exit__(self, *args, **kwargs):
        sys.setdlopenflags(self.old_flags)
