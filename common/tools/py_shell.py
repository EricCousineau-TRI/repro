#!/usr/bin/env python

# Start an arbitrary process in the background, such that it inherits Bazel's environment,
# especially for Python binaries that have complex Bazel-laced logic.
# TODO: Should this be --spawn_strategy=standalone?

import sys
import os
import subprocess

if len(sys.argv) > 1:
    args = sys.argv[1:]
    print "Execute: {}".format(args)
    subprocess.Popen(args)
else:
    # Print environment
    print "No arguments supplied!"
    print "usage: {} CMD...".format(os.path.basename(sys.argv[0]))
    exit(1)
