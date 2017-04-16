#!/bin/bash
toplevel=cpp_quick
target=tpl_inst

(
    cd ../bazel-bin/${toplevel}/_objs/${target}/${toplevel}
    set -x
    objdump -t -C tpl_inst_main.o
    objdump -t -C tpl_inst.o
) | tee dump.output.txt 2>&1
