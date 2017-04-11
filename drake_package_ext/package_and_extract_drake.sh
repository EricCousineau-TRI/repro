#!/bin/bash
### Wraps drake/tools/package_drake.sh such that it (a) provides artifacts in 
### a BUILD and INSTALL directory, and (b) will only update those artifacts
### only if they change.
### This mechanism is very unforgiving if you have a dirty Drake module.
set -e -u

# TODO(eric.cousineau): See if there is a mechanism to hash or dump a
# dependency to selectively trigger compilation via [c]make, rather than rely 
# on Git dirtyness
# TODO(eric.cousineau): Consider using rsync to simplify delta process.

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
drake_package_git=$drake_build/package-git
[[ -d $drake_package_git ]] || (
        mkcd $drake_package_git;
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
    echo "Prior build exists ($prev_git_status). Check against current ($cur_git_status)"
    if [[ -z $cur_git_status || -z $prev_git_status \
            || $cur_git_status = "dirty" || $prev_git_status = "dirty" ]]; then
        echo "Rebuild needed: current or previous build was dirty"
        need_rebuild=1
    elif [[ $cur_git_status != $prev_git_status ]]; then
        echo "Rebuild needed: current and previous build on different commits"
        need_rebuild=1
    else
        echo "No rebuild needed"
        need_rebuild=
    fi
else
    echo "First build"
fi

if [[ -z "$need_rebuild" ]]; then
    exit 0;
fi

echo "Generate package artifact from //drake/tools"
package=$drake_build/package.tar.gz
$drake/tools/package_drake.sh $package

cd $drake_package_git
# Remove prior files
rm -rf ./*

echo "Extract and commit current version"
tar xfz $package
git add -A :/
git commit --quiet -m \
    "Package artifacts for drake (status: $cur_git_status)" || {
        echo "No artifact difference detected." \
            "Skipping the rest of the rebuild.";
        echo "$cur_git_status" > $drake_git_status_file;
        exit 0;
    }

echo "Checkout and allow Git to handle deltas" \
    "  (be conservative on changing timestamps)"
git --work-tree=$install_dir checkout --quiet -f HEAD -- .

# On success, dump the git status
echo "$cur_git_status" > $drake_git_status_file
echo "Done"
