#!/bin/bash
set -eux

# Remove entrypoint symlinks.
rm ~/.local/bin/go ~/.local/bin/apptainer

# Remove install trees.
rm -rf ~/.local/opt/go ~/.local/opt/apptainer
