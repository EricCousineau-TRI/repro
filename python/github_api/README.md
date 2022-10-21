# GitHub API Stuff

Based loosely on
<https://github.com/RobotLocomotion/drake/blob/v1.9.0/tools/workspace/new_release.py>

Be sure to setup:

```sh
source ./setup.sh
```

## Create Token

Go to <https://github.com/settings/tokens>, and use classic token.
Beta token w/ granular access didn't seem to work at time of writing
(2022-10-21) :(

- Generate new token
- Check entire "repo" box.
- Create tokekn, then paste text into `~/.config/readwrite_github_api_token.txt`,
  or wherever.

## Move Issue

See `move_issue.py`.

## Preserve-then-Delete Branches

See `preserve_then_delete_branches.py`
