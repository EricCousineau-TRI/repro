#!/bin/bash
set -e -u

# @brief Generate a set of tests from a given Bazel project based on what files
# have changed on the current git diff

# TODO(eric.cousineau) Find if there is a simpler mechanism to query the set of
# targets dependent on a list of files.

# @ref https://bazel.build/versions/master/docs/query-how-to.html
# @ref https://bazel.build/versions/master/docs/query.html#rdeps

closure=//drake/...

branch=$(git rev-parse --abbrev-ref HEAD)
files=$(git diff --name-only $(git merge-base master $branch) $branch)
query=""

echo "Building query based on bazel targets..."
for file in $files; do
    # Super slow...
    target=$(bazel query $file)
    rdep="rdeps($closure, $target)"
    if [[ -z "$query" ]]; then
        query="$rdep"
    else
        query="$query union $rdep"
    fi
    echo "+ $file"
done

query="$query intersect kind(\"test\", $closure)"

echo "Execute query"
set -x
bazel query "$query"
