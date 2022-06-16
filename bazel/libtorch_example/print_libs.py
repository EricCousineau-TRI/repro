#!/usr/bin/env python

from subprocess import PIPE, run

def subshell(cmd, check=True):
    return run(cmd, shell=True, check=check, stdout=PIPE, text=True).stdout

libs = subshell("cd bazel-libtorch_example/external/libtorch/lib && ls *.so *.so.*").split()
libs.sort()
for lib in libs:
    print(f"    -l:{lib} \\")
