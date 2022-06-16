#!/bin/bash
set -eu
shopt -s expand_aliases

alias env-isolate='env -i HOME=$HOME DISPLAY=$DISPLAY SHELL=$SHELL TERM=$TERM USER=$USER PATH=/usr/local/bin:/usr/bin:/bin'
alias bash-isolate='env-isolate bash --norc'

bash-isolate "$@"
exit $?
