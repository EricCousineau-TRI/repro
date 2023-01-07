#!/bin/bash
set -eux

cd $(dirname ${BASH_SOURCE})

apptainer_bin=apptainer
# apptainer_bin=apptainer-v1.1.4  # Same
# apptainer_bin=apptainer-v1.0.0  # Same
# apptainer_bin=apptainer-v0.1.1  # Same

test_script=./glxgears_short.sh

${apptainer_bin} build --fakeroot --sandbox ./repro.sandbox ./repro.Apptainer

# Succeeds.
${apptainer_bin} exec ./repro.sandbox ${test_script}

# Fails.
${apptainer_bin} exec --nv ./repro.sandbox ${test_script}
