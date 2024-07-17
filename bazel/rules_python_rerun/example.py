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


def rerun_example():
    import numpy as np
    import rerun as rr

    rr.init("test", spawn=True)
    rr.spawn()

    SIZE = 10

    pos_grid = np.meshgrid(*[np.linspace(-10, 10, SIZE)]*3)
    positions = np.vstack([d.reshape(-1) for d in pos_grid]).T

    col_grid = np.meshgrid(*[np.linspace(0, 255, SIZE)]*3)
    colors = np.vstack([c.reshape(-1) for c in col_grid]).astype(np.uint8).T

    rr.log(
        "my_points",
        rr.Points3D(positions, colors=colors, radii=0.5)
    )


def main():
    print_paths()
    fix_rerun_path()
    rerun_example()


if __name__ == "__main__":
    main()
