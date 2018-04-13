#!/usr/bin/env python

import argparse
import os
import pickle
import sys
import subprocess

import os
from os.path import dirname, isdir, isfile


parser = argparse.ArgumentParser()
parser.add_argument("--version", type=str, default=None)
parser.add_argument("--python", type=str, default=None)
args = parser.parse_args()

def pwd():
    return os.getcwd()

def cd(d):
    if d:
        os.chdir(d)
    print("pwd: " + pwd())

def mkcd(d):
    if not os.path.exists(d):
        os.makedirs(d)
    cd(d)

def subshell(cmd):
    print("+ " + cmd)
    out = subprocess.check_output(cmd, shell=True, executable="/bin/bash").strip()
    print(" -> {}".format(out))
    return out

def call(cmd):
    print("+ " + cmd)
    subprocess.check_call(cmd, shell=True, executable="/bin/bash")

def call_status(cmd):
    print("+ " + cmd)
    return subprocess.call(cmd, shell=True, executable="/bin/bash")

def source(file):
    # Hoist environment post-sourcing of a bash file into this process.
    call("""
        source {};
        python -c 'import os, pickle; pickle.dump(dict(os.environ), open("env.pkl", "wb"))'
    """.format(file))
    with open('env.pkl', 'rb') as f:
        env = pickle.load(f)
        os.environ.update(env)

if args.python is None:
    python = subshell("which python")
else:
    python = args.path

if args.version is None:
    # Get numpy version from system python
    np_ver = subshell(python + " -c 'import numpy as np; print(np.version.version)'")
    np_git_rev = subshell(python + " -c 'import numpy as np; print(np.version.git_revision)'")
else:
    np_ver = args.version
    if np_ver[0] in "0123456789":
        np_git_rev = "v" + args.version
    else:
        np_git_rev = np_ver

fork = "https://github.com/numpy/numpy"
feature_commit_attempts = [
    'patches/feature_v1.11.0.patch',
    'patches/feature_v1.15.0.dev0+3ec8875.patch',
]
indicator_commit = 'patches/feature_indicator.patch'

# Clone fork
cd(dirname(__file__))
cur_dir = pwd()

def apply_patch_cmd(rel):
    patch_path = os.path.join(cur_dir, rel)
    return "git am --committer-date-is-author-date < {}".format(patch_path)

mkcd("tmp/patch_{}".format(np_ver))
if isdir("numpy"):
    cd("numpy")
    call("git clean -fxd")
    call("git checkout --force -B tmp_patch {}".format(np_git_rev))
else:
    call("git clone {}".format(fork))
    cd("numpy")
    call("git checkout -B tmp_patch {}".format(np_git_rev))

good = False
# Try out commits in order
for commit in feature_commit_attempts:
    if call_status(apply_patch_cmd(commit)) == 0:
        good = True
        break
    else:
        call("git am --abort")
if not good:
    raise Exception(
        "Could not apply any patch. Please try to resolve merge conflict.")

call(apply_patch_cmd(indicator_commit))

# Virtualenv
mkcd("../env")
if not isfile("bin/activate"):
    call("virtualenv --system-site-packages .")
source("bin/activate")
os.environ["NUMPY_PATCH"] = "1"

# Build
def has_patch():
    status = call_status("""
        python -c 'import numpy as np; has = "prefer_user_copyswap" in getattr(np.lib, "dev_patch_features", []); exit(not has)'
    """)
    return status == 0
cd("..")
if not has_patch():
    cd("numpy")
    call("python setup.py build -j 8 install")
    cd("..")
    # Test
    call("python -c 'import numpy; numpy.test()'")

# Test `pybind11`
pybind11 = "/home/eacousineau/proj/tri/repo/repro/externals/pybind11"
commit = "feature/numpy_dtype_user-wip"

if isdir("pybind11"):
    cd("pybind11")
    call("git checkout {}".format(commit))
    call("git clean -fxd")
else:
    call("git clone {} pybind11 -b {}".format(pybind11, commit))
    cd("pybind11")
mkcd("build")
call("cmake .. -DPYTHON_EXECUTABLE=$(which python)")
call("make -j pytest")
