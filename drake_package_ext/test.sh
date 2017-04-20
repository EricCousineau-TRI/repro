#!/bin/bash
set -e -u

suffix=${1-}
build_dir="build$suffix"
install_dir="$build_dir/install"

cmd() {
    ./package_and_extract_drake$suffix.sh $build_dir/ $install_dir
}

echo "[ First Build ]"
cmd

echo "[[ Suffix: $suffix ]]"
echo "[ Before ]"
# Check timestamps before/after
[[ -d $install_dir ]] && (
    cd $build_dir
    find . | xargs touch -h -t 201701010500
    ls -l install
    )

echo "[ During ]"
time cmd

echo "[ After ]"
ls -l $install_dir

echo -e "\n"
