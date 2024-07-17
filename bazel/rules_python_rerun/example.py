from pathlib import Path
import os
import re
import sys
import subprocess


def reformat_path(text):
    pattern = r".*\.runfiles"
    text = re.sub(pattern, "{runfiles}", text)
    text = text.replace(os.getcwd(), "${PWD}")
    return text


def print_paths():
    for p in sys.path:
        print(reformat_path(p))


def prepend_paths(var, ps):
    paths = os.environ.get(var, "").split(":")
    os.environ[var] = ":".join(ps + paths)


def fix_rerun_path():
    from bazel_tools.tools.python.runfiles import runfiles
    Rlocation = runfiles.Create().Rlocation
    bin = Path(Rlocation("rules_python_rerun/bin/rerun"))
    prepend_paths("PATH", [str(bin.parent)])


def main():
    print_paths()

    fix_rerun_path()

    import rerun as rr
    rr.init("test", spawn=True)
    rr.spawn()
    rr.set_time_sequence("frame", 42)


if __name__ == "__main__":
    main()
