#!/usr/bin/env python3

"""
Executes commands on both head and worker nodes.
"""

import argparse
import dataclasses as dc
from multiprocessing.pool import ThreadPool
from os.path import expanduser
import re
import shlex
from subprocess import run, PIPE, STDOUT
from textwrap import indent


def subshell(args, *, check=True):
    result = run(args, text=True, stdout=PIPE, stderr=STDOUT)
    if check and result.returncode != 0:
        print(result.stdout)
        raise RuntimeError(f"Error: {result.returncode}\n{args}")
    return result.stdout


@dc.dataclass
class Config:
    ssh_user: str
    ssh_key: str
    ssh_command: str
    ray_cluster: str
    process_count: int


def get_ray_ips(config, subcommand):
    text = subshell(["ray", subcommand, config.ray_cluster], check=False)
    if "Head node of cluster" in text and "not found" in text:
        return None
    return re.findall(r"^(\d+\.\d+\.\d+\.\d+)$", text, flags=re.MULTILINE)


def get_ray_head_and_worker_ips(config):
    # TODO(eric.cousineau): Find ray API for this.
    head_ips = get_ray_ips(config, "get-head-ip")
    if head_ips is None:
        return None
    assert len(head_ips) == 1
    worker_ips = get_ray_ips(config, "get-worker-ips")
    return head_ips + worker_ips


def bash_command_with_login(command):
    return shlex.join([
        "bash", "--login", "-c", command,
    ])


def ray_exec_all(config, host):
    # TODO(eric.cousineau): Find ray API for this.
    text = f"+ {config.ssh_command}\n"
    text += subshell([
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-i", config.ssh_key,
        "-o", "LogLevel=ERROR",
        f"{config.ssh_user}@{host}",
        bash_command_with_login(config.ssh_command),
    ])
    print(indent(text, f"[{host}] ").strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", type=str)
    args = parser.parse_args()

    config = Config(
        ssh_user="ubuntu",
        ssh_key="/path/to/key.pem",
        ssh_command=args.command,
        ray_cluster="./cluster.yaml",
        process_count=10,
    )

    ips = get_ray_head_and_worker_ips(config)
    if ips is None:
        print("Cluster not running")
        return
    print(f"ips: {ips}")
    with ThreadPool(config.process_count) as pool:
        func = lambda ip: ray_exec_all(config, ip)
        pool.map(func, ips)


if __name__ == "__main__":
    main()
