#!/bin/bash
### Wraps drake/tools/package_drake.sh such that it (a) provides artifacts in 
### a BUILD and INSTALL directory, and (b) will only update those artifacts
### only if they change.
### This mechanism is very unforgiving if you have a dirty Drake module.
set -e -u

# TODO(eric.cousineau): See if there is a mechanism to hash or dump a
# dependency to selectively trigger compilation via [c]make, rather than rely 
# on Git dirtyness

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

drake=$1
build_dir=$(mkcd $2 && pwd)
install_dir=$(mkcd $3 && pwd)

# HACK: Ensure that CMake's find_package will search PATH, detect ./bin/, 
# and resolve to parent directory to search in lib/cmake/
# See: https://cmake.org/cmake/help/v3.0/command/find_package.html
mkdir -p $install_dir/bin

drake_build=$build_dir/drake
mkdir -p $drake_build
# HACK: Use git-checkout to handle hashing artifacts and check if things
# need to be updated
# Will use one repository for staging, and then a second for "deploying"
# with minimal timestamp changes
drake_git=$drake_build/package-git
[[ -d $drake_git ]] || (
        mkcd $drake_git;
        git init --quiet .;
        echo "!*" > .gitignore # Ignore nothing, override user ~/.gitignore
    )

# Only rebuild if either (a) git was previously or currently dirty, or
# (b) if the SHAs do not match
git_ref() {
    # http://stackoverflow.com/a/5737794/7829525
    if [[ -n "$(git status --porcelain)" ]]; then
        echo "dirty"
    else
        git rev-parse --short HEAD
    fi
}
drake_git_status_file=$drake_build/drake-git-status
cur_git_status=$(cd $drake && git_ref)
need_rebuild=1
if [[ -e $drake_git_status_file ]]; then
    prev_git_status=$(cat $drake_git_status_file)
    if [[ -z $cur_git_status || -z $prev_git_status \
            || $cur_git_status = "dirty" || $prev_git_status = "dirty" ]]; then
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

# Generate package artifact from //drake/tools
package=$drake_build/package.tar.gz
$drake/tools/package_drake.sh $package

cd $drake_git
# Remove prior files
rm -rf ./*
# Extract and commit current version
tar xfz $package
git add -A :/
git commit --quiet -m \
    "Package artifacts for drake (status: $cur_git_status)" || {
        echo "No artifact difference detected." \
            "Skipping the rest of the rebuild.";
        echo "$cur_git_status" > $drake_git_status_file;
        exit 0;
    }

# Go to second repository and checkout to allow Git to handle deltas
drake_git_checkout=$drake_build/package-git-checkout
[[ -d "$drake_git_checkout" ]] || (
        git clone --quiet $drake_git $drake_git_checkout
    )
cd $drake_git_checkout
git pull --quiet origin master

# Symlink all non-hidden artifacts into the install directory
# NOTE: This will leave additional artifacts if the drake tree changes
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
                # NOTE: This will break if this script changes directory names
                # later on. Meh.
                error "Target file exists and is not a symlink: $target_file"
                exit 1
            fi
        else
            ln -s $source_dir/$rel_file $target_file
        fi
    done
}
files=$(find . -name '.git*' -prune -o -type f -print)
symlink_relative $drake_git_checkout $install_dir $files

# On success, dump the git status
echo "$cur_git_status" > $drake_git_status_file
