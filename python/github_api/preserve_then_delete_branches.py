import argparse
import os
import subprocess

import base


def shell(cmd, *, check=True, shell=True):
    print(f"+ {cmd}")
    return subprocess.run(cmd, shell=shell, check=check)


def is_same_repo(a, b):
    return a.url == b.url


def infer_ssh_url(url):
    base = "github.com"
    url = url.replace(f"https://{base}/", f"git@{base}:")
    if not url.endswith(".git"):
        url += ".git"
    return url


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

    skip_branches = {"main", "master"}

    branch_has_pr = []
    prs = repo.pull_requests(state="all")
    for pr in prs:
        if is_same_repo(pr.repository, repo):
            branch_name = pr.head.ref
            branch_has_pr.append(branch_name)

    https_url = repo.clone_url
    ssh_url = infer_ssh_url(https_url)

    tmp_dir = "/tmp/github_api_meh"
    os.makedirs(tmp_dir, exist_ok=True)
    os.chdir(tmp_dir)
    if not os.path.isdir(repo_name):
        print(ssh_url)
        shell(f"git clone {ssh_url}")
    os.chdir(repo_name)

    branches_to_pr = []
    for branch in repo.branches():
        if branch.name in skip_branches:
            continue
        if branch.name in branch_has_pr:
            continue
        if 
        branches_to_pr.append(branch)

    print("Branches to 
    for branch in cur_branches:
        print(branch.name)

    print("Press ENTER to open PR, close, then delete branch")
    input()

    for branch in cur_branches:


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
        print(f"Made PR: {new_pr}")

    print("Press ENTER to delete branches:")
    for branch in cur_branches:
        print(f"  {branch.name}")
    input()

    for branch in cur_branches:
        shell(f"git push origin :{branch.name}")


assert __name__ == "__main__"
main()
