#!/bin/bash
set -eu

# Extract the `[command]` portion of `sudo [opts] [command]`, then execute.

while [[ $# -gt 0 ]]; do
    case "$1" in
        -g|--group|-h|--host|-p|--prompt|-u|--user|-U|--other-user|-r|--role|-t|--type|-C|--close-from)
            # These have values.
            shift; shift;;
        -*)
            shift;;
        *)
            break;;
    esac
done

exec "$@"
