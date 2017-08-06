#!/bin/bash

source setup_target_env.sh //python/bindings/pymodule:type_binding

python -c "import pymodule.type_binding as tb; print('Module path:\n  {}'.format(tb.__file__))"
