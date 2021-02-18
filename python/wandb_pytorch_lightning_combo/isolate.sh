#!/bin/bash
set -eu

exec env -i \
    HOME=$HOME \
    DISPLAY=$DISPLAY \
    SHELL=$SHELL \
    TERM=$TERM \
    USER=$USER \
    PATH=/usr/local/bin:/usr/bin:/bin \
    LANG=$LANG \
    "$@"
