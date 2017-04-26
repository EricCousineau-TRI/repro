#!/bin/bash
set -x -e -u

suffix=${1-}
build_dir="build$suffix"
install_dir="$build_dir/install"

cmd() {
    if [[ $suffix = "_pr" ]]; then
        cd $DRAKE/tools
        mkdir -p $install_dir
        ./package_drake.sh -d $install_dir
    else
        ./package_and_extract_drake$suffix.sh $build_dir/ $install_dir > /dev/null 2> /dev/null
    fi
    echo "- Done"
}

echo "[[ Suffix: $suffix ]]"
echo "[ First Build ]"
cmd

echo "[ Before ]"
# Check timestamps before/after
find . | xargs touch -h -t 201701010500
before_file=$build_dir/before.txt
ls -l $install_dir | tee $before_file

echo "[ Second Build ]"
time cmd

echo "[ After ]"
after_file=$build_dir/after.txt
ls -l $install_dir | tee $after_file

echo "[ Diff ]"
git diff --no-index --word-diff $before_file $after_file || { echo "Difference!"; }

echo -e "\n\n"
