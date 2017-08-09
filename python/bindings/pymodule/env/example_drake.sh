#!/bin/bash
set -e -u

source_dir=$(cd $(dirname $BASH_SOURCE) && pwd)
cd ${DRAKE}/drake/bindings

source ${source_dir}/setup_target_env.sh //drake/bindings:pydrake

python -c "from pydrake.solvers import mathematicalprogram as mp; print('Module path:\n  {}'.format(mp.__file__))"
