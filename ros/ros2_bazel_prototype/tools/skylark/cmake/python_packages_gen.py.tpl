#!/usr/bin/env python3

"""
Generates a repository for using dist-packages.
"""

# Technique: Import all needed libraries, find all of the top-level modules,
# and symlink their folders in.

import glob
import os
from os.path import dirname, isdir, join
import sys

assert os.path.exists(args.path)

def main():
    sys.path.append(args.path)
    map(__import__, args.modules_exclude)
    modules_before = sys.modules.keys()
    map(__import__, args.modules)
    modules_new = set(sys.modules.keys()) - set(modules_before)

    os.mkdir("dist_packages")
    for module in modules_new:
        if "." in module:
            # Ignore anything but top-level modules.
            continue
        m = sys.modules[module]
        module_file = getattr(m, "__file__", None)
        if module_file is None:
            # May be a builtin.
            continue
        if not module_file.startswith(args.path):
            # Not part of this package.
            # TODO(eric.cousineau): Fail fast if this is not under something in the
            # initial `sys.path`?
            continue
        if "__init__.py" in module_file:
            # Handle for top-level modules.
            module_file = os.path.dirname(module_file)
        os.symlink(module_file, os.path.join("dist_packages", module))

    file_content = """
    py_library(
        name = "{name}",
        srcs = glob([
            "dist_packages/**/*.py",
        ]),
        data = glob([
            "dist_packages/**/*.so",
        ]),
        imports = ["dist_packages"],
        deps = {deps},
        visibility = ["//visibility:public"],
    )
    """.format(name=args.name, deps=args.deps).lstrip()

    with open('BUILD.bazel', 'w') as f:
        f.write(file_content)
