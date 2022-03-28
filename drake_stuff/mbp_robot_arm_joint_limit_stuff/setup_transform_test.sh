#!/bin/bash

# Either source this, or use it as a prefix:
#
#   source ./setup.sh
#   ./my_program
#
# or
#
#   ./setup.sh ./my_program

if [[ $# -lt "2" ]]; then
    echo "Please provide path to model directory and model file name."
    echo "      Usage:"
    echo "                  $bash model_transform.sh <model_directory_path> <model_file_name> "
    return 1
fi

source setup.sh

bash format_model_and_generate_manifest.sh "$1" "$2"

bash compare_model_via_drake_and_ingition_images.sh "$1" "$2"
