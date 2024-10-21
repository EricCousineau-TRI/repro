#!/bin/bash
set -eux

cd $(dirname ${BASH_SOURCE})

rm -rf repos && mkdir -p repos
cd repos

git init --bare main_remote.git
git init main_local

(
    cd main_local
    git lfs install
    git lfs track "*.bin"
    echo "Text" > README.md
    dd if=/dev/zero of=large-file.bin bs=1M count=10  # Create a dummy file
    git add .
    git commit -m "Initial commit"

    git config lfs.url http://localhost:8080
    git remote add main_remote ../main_remote.git
    git push main_remote
)

git init --bare mirror_remote.git

(
    GIT_LFS_SKIP_SMUDGE=1 git clone -o main_remote main_remote.git ./mirror_local
    git remote add mirror_remote ../mirror_remote.git
)
