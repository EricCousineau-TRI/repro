#!/bin/bash
set -e -u -x

cd $(dirname $BASH_SOURCE)

versions="1.11.0 1.12.0 1.13.0 1.14.0 1.14.1 master"

for version in $versions; do
    output="tmp/patch_${version}-output.txt"
    echo "Version ${version}: ${output}"
    {
        echo "Starting ${output}"
        python patchify.py --version=${version} > ${output} 2>&1 || echo "FAILED ${version}"
        echo "Finished ${output}"
    } &
done

echo "Waiting..."
wait
