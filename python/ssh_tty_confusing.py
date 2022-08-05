import os
import subprocess
from textwrap import dedent


def dumb_run(args, *, redirect):
    if redirect:
        out = subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
            text=True,
        )
        print(out.stdout)
    else:
        subprocess.run(args, check=True)


def main():
    # todo: need something that relies on "weird" tty behavior... like `bazel build` or `tshark` :(
    command = dedent(r"""
    tput sc
    for i in $(seq 5); do
        tput rc
        tput el
        echo -n "Step ${i}"
        sleep 0.1
    done
    tput ed
    echo
    """)
    dumb_run(["bash", "-c", command], redirect=False)
    dumb_run(["bash", "-c", command], redirect=True)
    user_host = os.environ["USER"] + "@localhost"
    ssh_args = [
        "ssh",
        "-tt",
        "-o",
        "BatchMode=yes",
        "-o",
        "LogLevel=ERROR",
        user_host,
    ]
    dumb_run(ssh_args + [command], redirect=False)
    dumb_run(ssh_args + [command], redirect=True)


assert __name__ == "__main__"
main()
