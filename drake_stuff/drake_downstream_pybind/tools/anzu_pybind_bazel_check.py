"""
Check for potentially risky usage of pybind macros in Bazel code.
"""

import argparse
import re
import sys
from textwrap import indent

REJECTED_PATTERNS = [
    r"\bpybind_py_library\b",
    r"\bdrake_pybind_py_library\b",
]

FAILURE_MESSAGE = """\
Direct usage of `pybind_py_library` and `drake_pybind_py_library` are
advised against, as it is easy to shoot yourself in the foot. Please use the
following macros instead to build Python bindings:
    load(
        "//tools/skylark:anzu_pybind.bzl",
        "anzu_cc_shared_library",
        "anzu_pybind_py_library",
    )
""".rstrip()


def eprint(*args):
    print(*args, file=sys.stderr)


def is_file_ok(filename):
    with open(filename, "r", encoding="utf8") as f:
        text = f.read()
    for pattern in REJECTED_PATTERNS:
        if re.search(pattern, text) is not None:
            return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", type=str, nargs="*")
    args = parser.parse_args()

    bad_files = []
    for file in args.files:
        if not is_file_ok(file):
            bad_files.append(file)

    if bad_files:
        eprint(f"The following files fail anzu_pybind_bazel_check:")
        eprint(indent("\n".join(sorted(bad_files)), "  "))
        eprint(FAILURE_MESSAGE)
        sys.exit(1)


if __name__ == "__main__":
    main()
