#!/usr/bin/env python2

import subprocess
import sys


# Determine if this is being run as a test, or via `./run`.
in_bazel = True
if "BAZEL_RUNFILES" in os.environ:
    # We are running via `./run`; we should be able to run without conversion.
    in_bazel = False
    print("Running direct notebook")
else:
    print("Running via conversion")



