import os
from os.path import dirname, realpath
from subprocess import run


def shell(cmd, *, verbose=False):
    if verbose:
        print(f"+ {cmd}")
    run(cmd, shell=True, check=True)


def run_with_pkg(pkg, *, find_links=False):
    shell("rm -rf ./venv")
    shell("python3 -m venv ./venv")
    shell("./venv/bin/pip install -U pip")
    if find_links:
        shell(f"./venv/bin/pip install {pkg} -f https://download.pytorch.org/whl/torch_stable.html", verbose=True)
    else:
        shell(f"./venv/bin/pip install {pkg}", verbose=True)
    # shell("./venv/bin/python ./torch_inv_bug.py")
    shell("./venv/bin/python ./torch_mm_bug.py")


def main():
    script_dir = dirname(realpath(__file__))
    os.chdir(script_dir)

    # On RTX 3090:

    # Fails
    run_with_pkg("torch==1.7.1+cu110", find_links=True)

    # Works
    run_with_pkg("torch==1.8.1+cu111", find_links=True)

    # Works
    run_with_pkg("torch==1.9.0+cu111", find_links=True)


assert __name__ == "__main__"
main()
