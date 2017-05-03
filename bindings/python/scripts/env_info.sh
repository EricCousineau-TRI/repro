#!/bin/bash

{
    bazel run //bindings:pydrake_type_binding_test \
        | sed \
            -e "s#$PATH#\${PATH}#g" -e "s#$LD_LIBRARY_PATH#\${LD_LIBRARY_PATH}#g" \
            -e "s#$PYTHONPATH#\${PYTHONPATH}#g" \
            -e "s#$USER#\${USER}#g" \
        | regex_sub.py \
            '/home/.*?/execroot/.*?/bin/' '${WORKSPACE}/bazel-bin/' -
             # From amber_developer_stack/common_scripts
} | tee env_info.output.txt
