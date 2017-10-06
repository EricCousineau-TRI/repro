#!/bin/bash

ARGS='-c dbg' source ../env/setup_target_env.sh //python/bindings/pymodule/tpl:${1-ownership_test}

python -c "import pymodule as tb; print('Module path:\n  {}'.format(tb.__file__))"
