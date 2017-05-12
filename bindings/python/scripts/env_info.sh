#!/bin/bash

{
    workspace_name=repro
    
    package=bindings
    import_path=${workspace_name}/${package}/python
    target=pydrake_type_binding_test
    runfiles="\${workspace}/bazel-bin/${package}/${target}.runfiles"

    bazel run //${package}:${target} \
        | sed \
            -e "s#$PATH#\${PATH}#g" \
            -e "s#$LD_LIBRARY_PATH#\${LD_LIBRARY_PATH}#g" \
            -e "s#$PYTHONPATH#\${PYTHONPATH}#g" \
            -e "s#$USER#\${USER}#g" \
        | regex_sub.py \
            '/home/.*?/execroot/.*?/bin/' '${workspace}/bazel-bin/' - \
        | sed \
            -e "s#$runfiles#\${runfiles}#g" \
            -e "s#$import_path#\${import_path}#g" \
            -e "s#$target#\${target}#g" \
            -e "s#$package#\${package}#g" \
            -e "s#$workspace_name#\${workspace_name}#g"
} | tee env_info.output.txt
# regex_sub: From amber_developer_stack/common_scripts
