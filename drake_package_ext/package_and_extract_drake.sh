#!/bin/bash
set -e -u -x
# TODO(eric.cousineau): See if there is a mechanism to hash or dump a dependency to selectively trigger compilation
# via (c)make

usage() {
    echo "Usage: $(basename $0) <DRAKE_DIR> <BUILD_DIR> <INSTALL_DIR>"
}
error() {
    echo "$@" >&2
}

[[ $# -eq 3 ]] || { error "Incorrect arguments specified"; usage; exit 1; }

mkcd() {
    mkdir -p $1 && cd $1;
}
git_is_dirty() {
    # http://stackoverflow.com/a/5737794/7829525
    test -n "$(git status --porcelain)"
}
git_ref() {
    if git_is_dirty; then
        echo "dirty"
    else
        git rev-parse --short HEAD
    fi
}
symlink_relative() {
    local source_dir=$1
    local target_dir=$2
    shift; shift;
    local rel_file
    for rel_file in $@; do
        local rel_dir=$(dirname $rel_file)
        local target_file=$target_dir/$rel_file
        mkdir -p $(dirname $target_file)
        if [[ -e $target_file ]]; then
            if [[ ! -L $target_file ]]; then
                # Target file is a symlink, it should be point to the same file
                # NOTE: This will break if this script changes directories. Meh.
                error "Target file exists and is not a symlink: $target_file"
                exit 1
            fi
        else
            ln -s $source_dir/$rel_file $target_file
        fi
    done
}

drake=$1
build_dir=$(mkcd $2 && pwd)
install_dir=$(mkcd $3 && pwd)

drake_build=$build_dir/drake
mkdir -p $drake_build
# HACK: Use git-checkout to handle hashing artifacts and check if things need to be updated
# Will use one repository for staging, and then a second for "deploying" with minimal timestamp changes
drake_git=$drake_build/package-git
if [[ ! -d $drake_git ]]; then
    (
        mkcd $drake_git;
        git init .;
        echo "!*" > .gitignore # Ignore nothing, override user ~/.gitignore
    )
fi

# Only rebuild if either (a) git was previously or currently dirty, or (b) if the SHAs do not match
# TODO(eric.cousineau) Find more robust mechanism
drake_git_status_file=$drake_build/status
cur_git_status=$(cd $drake && git_ref)
need_rebuild=1
if [[ -e $drake_git_status_file ]]; then
    prev_git_status=$(cat $drake_git_status_file)
    if [[ -z $cur_git_status || -z $prev_git_status || $cur_git_status = "dirty" || $prev_git_status = "dirty" ]]; then
        # Rebuild needed: was or is dirty
        need_rebuild=1
    elif [[ $cur_git_status != $prev_git_status ]]; then
        # Rebuild needed: different commits packaged
        need_rebuild=1
    else
        need_rebuild=
    fi
fi

if [[ -z "$need_rebuild" ]]; then
    echo "No rebuild needed"
    exit 0;
fi

# Generate package artifact
package=$drake_build/package.tar.gz
$drake/tools/package_drake.sh $package

cd $drake_git
# Remove prior files
rm -rf ./*
# Extract and commit current version
tar xfz $package
git add -A :/
git commit -m "Package artifacts for drake (status: $cur_git_status)"

# Go to second repository and checkout to allow Git to handle deltas
drake_git_checkout=$drake_build/package-git-checkout
[[ -d "$drake_git_checkout" ]] || ( git clone $drake_git $drake_git_checkout; )
cd $drake_git_checkout
git pull origin master

# Symlink all non-hidden artifacts into the install directory
# NOTE: This will leave additional artifacts if the drake tree changes
files=$(find . -name '.git*' -prune -o -type f -print)
symlink_relative $drake_git_checkout $install_dir $files
