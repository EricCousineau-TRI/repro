#!/bin/bash
set -eux

cd $(dirname ${BASH_SOURCE})

rm -rf ./tmp
mkdir ./tmp && cd ./tmp

git clone git@github.com:EricCousineau-TRI/test_github_api.git
cd test_github_api

# Delete all remote branches.
remote_branches=$(git branch --list -r 'origin/*' --format '%(refname)' | sed 's#refs/remotes/origin/##g')
for branch in ${remote_branches}; do
    git push origin :${branch} || :
done

{ git checkout --detach && git branch -D main ; } || :

# Now create three branches.

git checkout --orphan main
git reset && rm -rf ./*

echo "a" >> README
git add ./README && git commit -m "Commit"
git push -f origin main

git checkout -B branch-1
echo "a" >> README
git add ./README && git commit -m "Commit"
# Not merged.
git push origin branch-1

git checkout -B branch-2 main
git push origin branch-2
