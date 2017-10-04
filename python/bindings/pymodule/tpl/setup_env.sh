#!/bin/bash

source ../env/setup_target_env.sh //python/bindings/pymodule/tpl:scalar_type_test

python -c "import pymodule as tb; print('Module path:\n  {}'.format(tb.__file__))"
