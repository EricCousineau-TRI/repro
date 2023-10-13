import os
import subprocess


def shell(cmd, check=True):
    print(f"+ {cmd}")
    return subprocess.run(cmd, shell=True, check=check)


def cherry_pick(proto_commit):
    shell(f"git cherry-pick {proto_commit}")


def replay_merge(proto_commit, main_commit):
    # git rebase --preserve-merges no longer exists
    # git rebase --rebase-merges didn't seem to (easily) work
    # git cherry-pick -m 1 didn't seem to preserve the actual merge history
    result = shell(f"git merge {main_commit}", check=False)
    if result.returncode != 0:
        # shell("git reset")
        shell(f"git checkout {proto_commit} -- :/")
        shell("git gui")


# resolved, mainline   OR
# cherry


replay_commits="""
94bc246d  937161476
e7881078  v2.9.1
5951b248
f0a4dd50  ec24786eab~
dbed3e4b
a022f1bc  ec24786eab
6bc74137
773f343f  v2.10.0
224d0b31
cc8f36ea
""".strip().splitlines()


def main():
    os.chdir("/tmp/pybind11")
    shell("git reset --hard 11284cc442c9")

    for line in replay_commits:
        pieces = line.split()
        if len(pieces) == 1:
            commit, = pieces
            cherry_pick(commit)
        elif len(pieces) == 2:
            proto, main = pieces
            replay_merge(proto, main)


assert __name__ == "__main__"
main()
