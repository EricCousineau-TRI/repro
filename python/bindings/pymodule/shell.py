#!/usr/bin/env python

# Start an arbitrary process in the background, such that it inherits Bazel's environment.
# TODO: Should this be --spawn_strategy=standalone?

import sys
import subprocess

args = sys.argv[1:]
print "Execute: {}".format(args)
subprocess.Popen(args)
