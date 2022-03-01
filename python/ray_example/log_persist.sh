#!/bin/bash
set -eux

# HACK: Save ray logs
# TODO(eric): Unclear how to change logging directory?
# https://docs.ray.io/en/releases-1.9.2/ray-logging.html#logging-directory-structure
old_log_dir=/tmp/ray
new_log_dir=~/ray-logs
if [[ ! -L ${old_log_dir} ]]; then
    if [[ -d ${old_log_dir} && -d ${new_log_dir} ]]; then
        echo "Shouldn't already exist!" >&2
        exit 1
    fi

    if [[ ! -d ${old_log_dir} ]]; then
        mkdir -p ${old_log_dir}
    fi

    if [[ -d ${old_log_dir} ]]; then
        mv ${old_log_dir} ${new_log_dir}
    else
        mkdir -p ${new_log_dir}
    fi

    ln -s ${new_log_dir} ${old_log_dir}
fi
