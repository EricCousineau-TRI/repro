#!/bin/bash
set -e -u

suffix=${1-}

echo "[[ Suffix: $suffix ]]"
echo "[ Before ]"
# Check timestamps before/after
[[ -d build$suffix ]] && (
    cd build$suffix
    find . | xargs touch -h -t 201701010500
    ls -l install
    )

echo "[ During ]"
time ./package_and_extract_drake$suffix.sh $DRAKE build$suffix/ build$suffix/install

echo "[ After ]"
ls -l build$suffix/install
