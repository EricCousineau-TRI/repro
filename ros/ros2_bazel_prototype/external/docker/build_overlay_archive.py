#!/usr/bin/env python3

import argparse
import os
from os.path import basename, dirname, isdir, join
from shutil import rmtree
import subprocess
import sys


def run(args, **kwargs):
    if isinstance(args, list):
        cmd = " ".join(args)
    else:
        cmd = args
    print("\n + {}".format(cmd))
    return subprocess.run(args, check=True, **kwargs)


def direct_main(_):
    mnt = os.getcwd()
    if isdir("gen"):
        rmtree("gen")
    os.mkdir("gen")
    os.chdir("gen")
    os.mkdir("src")
    run("vcs import src < {}/overlay.repo".format(mnt), shell=True)
    run(["bash", "-c", "source /opt/ros/crystal/setup.bash \
        && colcon build --merge-install \
            --cmake-args -DBUILD_TESTING=OFF"])
    output_file = join(mnt, "overlay.tar.bz2")
    run(["tar", "-cjf", output_file, "."], cwd="./install")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_direct", action="store_true")
    args = parser.parse_args()

    sys.stdout = sys.stderr
    os.chdir(dirname(__file__))

    if not args.is_direct:
        # Build container.
        run([
            "docker", "build", "-t", "ros2_local:crystal",
            "-f", "./ros2-apt-crystal.Dockerfile", ".",
        ])
        # Execute this script.
        run([
            "docker", "run",
            # To remove annoying user mismatch:
            # https://stackoverflow.com/a/45959754/7829525
            "-v", "/etc/passwd:/etc/passwd",
            "-u", "{}:{}".format(os.getuid(), os.getgid()),
            "--rm",
            "-v", "{}:/mnt".format(os.getcwd()),
            "ros2_local:crystal",
            "python3", "/mnt/" + basename(__file__),
            "--is_direct",
        ])
    else:
        direct_main(args)


if __name__ == "__main__":
    main()
