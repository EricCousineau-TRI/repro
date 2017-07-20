#!/bin/bash

bazel clean
bazel test --test_output=all --action_env="LCM_DEFAULT_URL=udpm://239.255.67.70:7668?ttl=0" :main | \
    sed "
        s#$PATH#path#g
        s#$LD_LIBRARY_PATH#ld_library_path#g
        s#$HOME#/home#g
        s#$USER#user#g
        " | tee output.txt
