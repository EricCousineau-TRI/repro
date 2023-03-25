#!/usr/bin/env python3

"""
Given a URL (or a pattern that should expand to a URL), downloads the file,
computes the sha256sum, and places it inside of Bazel's content-addressable
cache.

Examples:

* Full URL - GitHub's download will parse the HEAD part and give you the
  redirect with latest commit.

    ./bazel_hash_and_cache.py \
        https://github.com/RobotLocomotion/drake/archive/HEAD.tar.gz

* Pattern for Drake source

    ./bazel_hash_and_cache.py drake@HEAD

* Pattern for drake-ros source

    ./bazel_hash_and_cache.py drake-ros@main

* Pattern for Drake nightly

    ./bazel_hash_and_cache.py drake-nightly@latest
    ./bazel_hash_and_cache.py drake-nightly@2020603

"""

# TODO(eric.cousineau): Also should tinker with this?
# https://stackoverflow.com/a/60952814/7829525 - clone log only.

import argparse
from os import mkdir
from os.path import realpath, dirname, isdir, getsize, join
import re
from shutil import move
from subprocess import check_call, check_output
import sys
from tempfile import mkdtemp
from textwrap import indent

_size_map = {'K': 1, 'M': 2, 'G': 3}


def parse_size_bytes(s):
    s = s.upper()
    if s.endswith('B'):
        s = s[:-1]
    suffix = s[-1]
    num = float(s[:-1])
    exp = _size_map[suffix]
    div = 1024**exp
    return num * div


# Substituted in order.
_URL_PATTERNS = (
    (r"drake@([\w\.]+)",
        r"https://github.com/RobotLocomotion/drake/archive/\1.tar.gz"),
    (r"drake-ros@([\w\.]+)",
        r"https://github.com/RobotLocomotion/drake-ros/archive/refs/heads/\1.tar.gz"),
    (r"drake-nightly@([\w\.]+)",
        r"https://drake-packages.csail.mit.edu/drake/nightly/drake-\1-focal.tar.gz"),  # noqa
    (r"pybind11@([\w.]+)",
        r"https://github.com/RobotLocomotion/pybind11/archive/\1.tar.gz"),
    (r"([\w\-\./]+)@([\w.]+)",
        r"https://github.com/\1/archive/\2.tar.gz"),
    (r"(https?://.*)",
        r"\1"),
)


def resolve_url(url):
    checked = []
    for pattern, repl in _URL_PATTERNS:
        pattern = f'^{pattern}$'
        new_url, count = re.subn(pattern, repl, url)
        if count > 0:
            print(f"Matched pattern: {pattern}")
            return new_url
        checked.append(pattern)
    else:
        print(f"Unaccepted pattern: {url}")
        print(f"Patterns checked:")
        print(indent("\n".join(checked), "  "))
        sys.exit(1)


def get_sha256_file_func(bazel_repository_cache):
    # Made as function factory to fail fast (not after downloading).
    # Query bazel if necessary.
    if bazel_repository_cache is None:
        script_dir = dirname(realpath(__file__))
        fake_bazel_workspace = join(script_dir, "fake_bazel_workspace")
        bazel_repository_cache = check_output(
            ["bazel", "info", "repository_cache"],
            cwd=fake_bazel_workspace, encoding="utf8").strip()
        check_output(["bazel", "shutdown"], cwd=fake_bazel_workspace)
    sha256_dir = join(bazel_repository_cache, "content_addressable", "sha256")
    if not isdir(sha256_dir):
        print(f"Does not exist: {sha256_dir}")
        sys.exit(1)

    def get_sha256_file(sha256):
        assert len(sha256) == 64
        sha256_file = join(sha256_dir, sha256, "file")
        return sha256_file

    return get_sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "url", type=str,
        help="URL (or pattern) to download, hash, and cache")
    parser.add_argument(
        "--min_size", type=str, default="10K", help="Sanity check")
    parser.add_argument(
        "--bazel_repository_cache", type=str, default=None,
        help="Location of repo cache. If not specified, will query.")
    parser.add_argument(
        "--show_file", action="store_true",
        help="Interpret URL as sha256 and show its local cache path.")
    args = parser.parse_args()

    get_sha256_file = get_sha256_file_func(args.bazel_repository_cache)

    if not args.show_file:
        tmp_dir = mkdtemp()
        tmp_file = join(tmp_dir, "file")
        url = resolve_url(args.url)
        print(f"Fetching URL: {url}")
        check_call(["wget", url, "-O", tmp_file])
        # Ensure it's of minimum size (e.g. avoid using 404 error downloads).
        file_size = getsize(tmp_file)
        min_size = parse_size_bytes(args.min_size)
        if file_size < min_size:
            print(f"File too small: {file_size} < {min_size}")
            sys.exit(1)
        sha256, _ = check_output(
            ["sha256sum", tmp_file], encoding="utf8").strip().split()
        sha256_file = get_sha256_file(sha256)
        mkdir(dirname(sha256_file))
        move(tmp_file, sha256_file)
        print()
        print(f"Downloaded: {sha256_file}")
        print()
        print(f"sha256: {sha256}")
        print()
    else:
        assert args.show_file is not None
        sha256 = args.url
        sha256_file = get_sha256_file(sha256)
        print(f"Downloaded: {sha256_file}")
        print()


if __name__ == "__main__":
    main()
