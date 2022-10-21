# GitHub API Stuff

Based loosely on
<https://github.com/RobotLocomotion/drake/blob/v1.9.0/tools/workspace/new_release.py>

Be sure to setup:

```sh
source ./setup.sh
```

## Create Token

Go to <https://github.com/settings/tokens?type=beta>, and use fine-grained
permissions:

- Generate new token
- Do "Only select repositories", give it only access to your relevant repository.
- Repository permissions:
    - Issues: "Read and write"
    - Pull requests: "Read and write"
- Create tokekn, then paste text into `~/.config/readwrite_github_api_token.txt`,
  or wherever.

## Move Issue

See `move_issue.py`.

## Preserve-then-Delete Branches

See `preserve_then_delete_branches.py`
