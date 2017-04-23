#!/bin/bash

# Port changes layered on one file for a set of commits, and move them to another (possibly) new file, while
# maintaining the original source file prior to changes.

# Objective: 
# Given modifications to an existing file along a branch (that has not yet been pushed), rebase the changes
# To start with a new file and layer the changes on that file
set -x -e -u

base=origin/master
rebasee=feature/xpr_tpl_attempt
new_branch=redux

src=./matrix_stack.cc
tgt=./matrix_stack_xpr_tpl.cc

ancestor=$(git merge-base $base $rebasee)

shas="$(git rev-list $ancestor..$rebasee | tac)"

# http://stackoverflow.com/questions/3357280/print-commit-message-of-a-given-commit-in-git

git checkout -B $new_branch $ancestor
first=1
for sha in $shas; do
    msg="$(git show -s --format=%B $sha)"
    git cat-file blob $sha:$src > $tgt
    git add $tgt
    if [[ -n $first ]]; then
        echo "This is the first commit."
        echo "Please make your intermediate changes, then press [ENTER]."
        read something
        first=
        git add -A .
    fi
    git commit -m "$msg"
done
