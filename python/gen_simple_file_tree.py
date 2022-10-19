#!/usr/bin/env python

r"""
Generates tree of files for testing throughput on NAS stuff, like EFS, Luster,
etc.

Example usage:

    $ rm -rf /tmp/gen_test
    $ ./gen_big_file_tree.py --output_dir=/tmp/gen_test --num_files=10000
    $ du -hs /tmp/gen_test
    978M    /tmp/gen_test
    $ find /tmp/gen_test -type f | wc -l
    10000

"""

# TODO(eric.cousineau): Currently, this mixes deterministic names with
# non-deterministic file content. Should sync, and add options for fully
# deterministic or (more) random.

import argparse
import hashlib
import os
import sys

_size_map = {'K': 1, 'M': 2, 'G': 3}


def parse_size_bytes(s):
    s = s.upper()
    if s.endswith('B'):
        s = s[:-1]
    suffix = s[-1]
    num = float(s[:-1])
    exp = _size_map[suffix]
    div = 1024**exp
    return int(num * div)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def int_to_sha256(x):
    data = bytes(str(x), encoding="utf8")
    return sha256(data)


def write_file_of_random_bytes(filename, size):
    # https://stackoverflow.com/a/14276423/7829525
    # For this purpose, we don't really care about crypotographic security. Oh
    # well.
    data = os.urandom(size)
    assert len(data) == size
    with open(filename, "wb") as f:
        f.write(data)


def expand_hash_to_subdirs(base, num_subdirs):
    pieces = []
    for i in range(num_subdirs):
        prefix = base[:i + 1]
        pieces.append(prefix)
    pieces.append(base)
    return "/".join(pieces)


def main():
    parser = argparse.ArgumentParser()
    # N.B. ext4 and `ls` seems to only show 4K increments in size.
    parser.add_argument(
        "--size", type=str, default="100K",
        help="Size of each file to emit.",
    )
    parser.add_argument(
        "--num_files", type=int, default=10,
        help="How many files to create.",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory. Must not exist.",
    )
    parser.add_argument(
        "--num_subdirs", type=int, default=0,
        help="Number of subdirectories (using portion of random hash).",
    )
    args = parser.parse_args()

    size = parse_size_bytes(args.size)
    if os.path.exists(args.output_dir):
        print("--output_dir must not exist.", file=sys.stderr)
        sys.exit(1)

    for i in range(args.num_files):
        sha = int_to_sha256(i)
        base = expand_hash_to_subdirs(sha, args.num_subdirs)
        filename = os.path.join(args.output_dir, base)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        write_file_of_random_bytes(filename, size)

    print("[ Done ]")


if __name__ == "__main__":
    main()
