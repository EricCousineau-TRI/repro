import os
import subprocess
from textwrap import dedent


def dumb_run(args, *, shell):
    out = subprocess.run(
        args,
        shell=shell,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
        text=True,
    )
    print(out.stdout)


def main():
    # todo: need something that relies on "weird" tty behavior... like `bazel build` or `tshark` :(
    command = dedent("""
    watch -n 0.5 -g date
    """)
    dumb_run(command, shell=True)
    user_host = os.environ["USER"] + "@localhost"
    dumb_run(
        [
            "ssh",
            "-tt",
            "-o",
            "BatchMode=yes",
            "-o",
            "LogLevel=ERROR",
            user_host,
            command,
        ],
        shell=False,
    )


assert __name__ == "__main__"
main()
