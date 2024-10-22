#!/bin/bash
set -eux

cd $(dirname ${BASH_SOURCE})

export EDITOR=true

# Clear lfs content.
rm -rf lfs-content/*
rm -f lfs.db

rm -rf repos && mkdir -p repos
cd repos

make-dummy-mb() { (
    set +x
    file=${1}
    size_mb=${2}
    dd if=/dev/zero of=${file} bs=1M count=${size_mb}  # Create a dummy file
) }

git init --bare main_remote.git
git init main_checkout

(
    cd main_checkout
    git lfs install
    git lfs track "*.bin"
    echo "Text" > README.md
    make-dummy-mb large-file.bin 10

    git add .
    git commit -m "Initial commit"

    git config lfs.url http://localhost:8080
    git remote add origin ../main_remote.git
    git push origin
)

# secondary fork, pushes change
(
    git clone ${PWD}/main_remote.git second_checkout \
        --config lfs.url=http://localhost:8080
    cd second_checkout
    # make second large file
    make-dummy-mb large-file-2.bin 20
    git add .
    git commit -m "new commit"
    git push origin
)

(
    cd main_checkout
    echo "Other file" > new_file.txt
    git add .
    git commit -m "second file"

    # Fetch.
    git fetch origin

    # this will fail
    git rebase origin/master
)
