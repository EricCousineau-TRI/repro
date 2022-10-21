import argparse

import base


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

    prs = repo.pull_requests(state="all")
    pr_branches = [pr.head for pr in prs]
    print(pr_branches)

    skip_branches = {"main"}

    for branch in repo.branches():
        if branch.name in skip_branches:
            continue
        print(branch.name)


assert __name__ == "__main__"
main()
