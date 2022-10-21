import os

import github3


def add_parser(parser):
    parser.add_argument(
        "--token_file", default="~/.config/readwrite_github_api_token.txt",
        help="Uses an API token read from this filename",
    )


def login(args):
    with open(os.path.expanduser(args.token_file), "r") as f:
        token = f.read().strip()
    gh = github3.login(token=token)
    return gh


def parse_repo(repo, *, tail=False):
    owner, repo_name = repo.split("/")
    if tail:
        repo_name, num = repo_name.split("#")
        return owner, repo_name, num
    else:
        return owner, repo_name
