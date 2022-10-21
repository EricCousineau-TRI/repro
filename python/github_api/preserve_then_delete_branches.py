#!/usr/bin/env python

"""
Preserves repo branches by opening a PR against, them, closing PR, then deleting
said branch.

Example:
    ./preserve_then_delete_branches.py --repo EricCousineau-TRI/test_github_api
"""

import argparse
import os
import subprocess

import base


def shell(cmd, *, check=True, shell=True):
    print(f"+ {cmd}")
    return subprocess.run(cmd, shell=shell, check=check)


def subshell(cmd):
    print(f"+ $({cmd})")
    result = subprocess.run(
        cmd, shell=True, check=True, text=True, stdout=subprocess.PIPE
    )
    return result.stdout.strip()


def is_same_repo(a, b):
    return a.url == b.url


def infer_ssh_url(url):
    base = "github.com"
    url = url.replace(f"https://{base}/", f"git@{base}:")
    if not url.endswith(".git"):
        url += ".git"
    return url


def is_branch_merged(branch, main):
    remote = "origin"
    branch_ref = f"{remote}/{branch}"
    main_ref = f"{remote}/{main}"
    # This seems to work when both commits are the same too.
    result = shell(
        f"git merge-base --is-ancestor {branch_ref} {main_ref}",
        check=False,
    )
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser()
    base.add_parser(parser)
    parser.add_argument(
        "--repo", type=str, required=True,
        help="Form of {owner}/{repo_name}",
    )
    parser.add_argument("-n", "--dry_run")
    args = parser.parse_args()

    gh = base.login(args)
    owner, repo_name = base.parse_repo(args.repo)
    repo = gh.repository(owner, repo_name)

    main_branch = "main"
    skip_branches = {main_branch}

    branches_with_open_pr = []
    branches_with_closed_pr = []
    prs = repo.pull_requests(state="all")
    for pr in prs:
        if is_same_repo(pr.repository, repo):
            branch_name = pr.head.ref
            if pr.state == "open":
                branches_with_open_pr.append(branch_name)
            elif pr.state == "closed":
                branches_with_closed_pr.append(branch_name)

    https_url = repo.clone_url
    ssh_url = infer_ssh_url(https_url)

    # TODO(eric.cousineau): I dunno how to make github3.py do this.
    tmp_dir = "/tmp/github_api_meh"
    os.makedirs(tmp_dir, exist_ok=True)
    os.chdir(tmp_dir)
    if not os.path.isdir(repo_name):
        print(ssh_url)
        shell(f"git clone {ssh_url}")
    os.chdir(repo_name)
    shell("git fetch origin")

    branches_to_delete = []
    branches_to_pr = []
    for branch in repo.branches():
        if branch.name in skip_branches:
            continue
        if branch.name in branches_with_open_pr:
            continue

        branches_to_delete.append(branch)

        if (
            branch.name not in branches_with_closed_pr and
            not is_branch_merged(branch.name, main_branch)
        ):
            branches_to_pr.append(branch)

    if len(branches_to_pr) > 0:
        print("Branches to open / close PR")
        for branch in branches_to_pr:
            print(f"  {branch.name}")

        print()
        print("Press ENTER to continue")
        input()

        for branch in branches_to_pr:
            new_pr = repo.create_pull(
                title=f"Preserve branch '{branch.name}'",
                base="main",
                head=branch.name,
                body=(
                    f"This is an automated PR to preserve {branch.name} before "
                    f"deleting it."
                ),
            )
            new_pr.close()
            print(f"  Opened/Closed PR: {new_pr.url}")

    if len(branches_to_delete) > 0:
        print("Branches to delete:")
        for branch in branches_to_delete:
            print(f"  {branch.name}")

        print()
        print("Press ENTER to continue")
        input()

        for branch in branches_to_delete:
            # TODO(eric.cousineau): How to use github API?
            shell(f"git push origin :{branch.name}")

    print("[ Done ]")


assert __name__ == "__main__"
try:
    main()
except KeyboardInterrupt:
    pass
