#!/bin/bash

# Goal: Process a patch and show new lines that are copy/pastable

sed \
    -e 's#^ #|#g' \
    -e 's#^+##g'
