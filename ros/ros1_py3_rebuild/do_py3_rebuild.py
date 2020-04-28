#!/usr/bin/env python3

"""
Updates neighboring archive using Docker build.
"""

import argparse
import os
from os.path import abspath, basename, dirname, isdir, join
from shutil import rmtree, copytree, ignore_patterns
import subprocess
import sys

from module_list import PY3_REBUILD

ROS_ROOT = "/opt/ros/melodic"
ROS_BASH = [
    "bash", "-c", f"source {ROS_ROOT}/setup.bash; exec \"$@\"", "--"]


def run(args, **kwargs):
    print("\n + {}".format(args))
    return subprocess.run(args, check=True, **kwargs)


def direct_main(script_dir):
    build_dir = join(script_dir, "build")
    if isdir(build_dir):
        rmtree(build_dir)
    os.makedirs(build_dir)
    os.chdir(build_dir)
    os.makedirs("src")
    run(["vcs", "import", "./src",
         "--input", join(script_dir, "do_py3_rebuild.repo")])
    # TODO(eric.cousineau): I can't figure out what the right combination of
    # using `--pkg` vs. `--only-pkg-with-deps` such that it doesn't
    # unnecessarily build non-Python deps and it configures correctly.
    # If I use `--pkg` but something like `roslz4` is in a workspace with other
    # packages, it seems to fail. If I use `--only-pkg-with-deps`, the build of
    # `tf2_py` will cause a whole slew of packages to be built.
    parent_and_subdirs = [
        ("src/vision_opencv", "cv_bridge"),
        ("src/geometry2", "tf2_py"),
    ]
    for parent, subdir in parent_and_subdirs:
        copytree(join(parent, subdir), join("src", basename(subdir)))
        rmtree(parent)
    # TODO(eric.cousineau): Build with `colcon`.
    run(ROS_BASH + [
        "catkin_make_isolated",
            "-DBUILD_TESTING=OFF",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DPYTHON_EXECUTABLE=/usr/bin/python3",
            f"-DCMAKE_INSTALL_RPATH={ROS_ROOT}/lib",
            "--install",
            "--only-pkg-with-deps"] + PY3_REBUILD)
    dist_dir = "install_isolated/lib/python3/dist-packages"
    assert isdir(dist_dir), dist_dir
    results_dir = "results"
    os.makedirs(results_dir)
    for pkg in PY3_REBUILD:
        copytree(
            join(dist_dir, pkg), join(results_dir, pkg),
            ignore=ignore_patterns("__pycache__"))
    archive = join(script_dir, "py3_rebuild.tar.gz")
    run(["tar", "-czf", archive, "."], cwd=results_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--is_direct", action="store_true",
        help="Build directly. This is generally not used on the host, but "
             "instead in the Docker container.")
    args = parser.parse_args()

    sys.stdout = sys.stderr
    script_dir = dirname(abspath(__file__))

    if args.is_direct:
        # Execute directly.
        direct_main(script_dir)
    else:
        os.chdir(script_dir)
        image = "repro/do_py3_rebuild"
        script_in_docker = join("/mnt", basename(__file__))
        run(["docker", "build", "-t", image,
             "-f", "do_py3_rebuild.Dockerfile", "."])
        # Execute this script.
        run([
            "docker", "run",
            # To remove annoying user mismatch:
            # https://stackoverflow.com/a/45959754/7829525
            "-v", "/etc/passwd:/etc/passwd",
            "-u", "{}:{}".format(os.getuid(), os.getgid()),
            "--rm",
            "-v", "{}:/mnt".format(script_dir),
            image,
            "python3", script_in_docker, "--is_direct",
        ])


if __name__ == "__main__":
    main()
