#!/bin/bash

bazel test --test_output=all :main | \
    sed "
        s#$PATH#path#g
        s#$LD_LIBRARY_PATH#ld_library_path#g
        s#$HOME#/home#g
        s#$USER#user#g
        " | tee output.txt
