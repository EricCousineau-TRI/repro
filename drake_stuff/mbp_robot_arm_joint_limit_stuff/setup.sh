#!/bin/bash

# Either source this, or use it as a prefix:
#
#   source ./setup.sh
#   ./my_program
#
# or
#
#   ./setup.sh ./my_program

_cur_dir=$(cd $(dirname ${BASH_SOURCE}) && pwd)
_venv_dir=${_cur_dir}/venv

_download_drake() { (
    # See: https://drake.mit.edu/from_binary.html
    # Download and echo path to stdout for capture.
    set -eux

    base=drake-latest-focal.tar.gz
    dir=~/Downloads
    uri=https://drake-packages.csail.mit.edu/drake/nightly
    if [[ ! -f ${dir}/${base} ]]; then
        wget ${uri}/${base} -O ${dir}/${base}
    fi
    echo ${dir}/${base}
) }

_build_gazebo_plugin() { (

    # We need to generate the gazebo plugin for
    # IoU testing of joints
    cd $1
    git clone https://github.com/marcoag/ModelPhotoShoot.git
    cd ModelPhotoShoot
    mkdir build
    cd build
    cmake ..
    make

) }

_test_models() { (

    # Generate model pics using the gazebo plugin

    cd "$3"
    mkdir -p visual/model/
    cp -r "$1"/* visual/model/
    cd visual/model/
    sed -i 's/'"'"'/"/g' "$2"
    MODEL_NAME=`grep 'model name' model.sdf|cut -f2 -d '"'`
    sed -i "s,model://$MODEL_NAME/,,g" "$2"
    # Default pose pics:
    mkdir -p $3/visual/pics/default_pose/
    GAZEBO_MODEL_PATH="$3/visual/model/" GAZEBO_PLUGIN_PATH=$3/ModelPhotoShoot/build/:$GAZEBO_PLUGIN_PATH gzserver -s libmodelphotoshoot.so worlds/blank.world --propshop-save $3/visual/pics/default_pose/ --propshop-model "$3/visual/model/$2" --data-file $3/visual/pics/default_pose/poses.txt
    # Random joint positions pics:
    mkdir -p $3/visual/pics/random_pose/
    GAZEBO_MODEL_PATH="$3/visual/model/" GAZEBO_PLUGIN_PATH=$3/ModelPhotoShoot/build/:$GAZEBO_PLUGIN_PATH gzserver -s libmodelphotoshoot.so worlds/blank.world --propshop-save $3/visual/pics/random_pose/ --propshop-model "$3/visual/model/$2" --data-file $3/visual/pics/random_pose/poses.txt --random-joints

    #Generate model pics using drake, run 
    #IoU tests and extra checks
    cd ${_cur_dir}
    ./test_models.py "$1" "$2" "$3/visual/"

    # Test collision part:
    # Create a model that exchanges visual and collision
    # meshes in order to take pics of the collision mesh
    # and compare them
    cd "$3"
    mkdir -p collisions/model/
    cp -r "$1"/* collisions/model/
    cd collisions/model/
    find . -name "*.mtl" | xargs rm
    sed -i 's/collision/temporalname/g' "$2"
    sed -i 's/visual/ignore:collision/g' "$2"
    sed -i 's/temporalname/visual/g' "$2"

    sed -i 's/'"'"'/"/g' "$2"
    sed -i "s,model://$MODEL_NAME/,,g" "$2"

    # Default pose pics:
    mkdir -p $3/collisions/pics/default_pose/
    GAZEBO_MODEL_PATH="$3/collisions/model/" GAZEBO_PLUGIN_PATH=$3/ModelPhotoShoot/build/:$GAZEBO_PLUGIN_PATH gzserver -s libmodelphotoshoot.so worlds/blank.world --propshop-save "$3/collisions/pics/default_pose/" --propshop-model "$3/collisions/model/$2" --data-file "$3/collisions/pics/default_pose/poses.txt"

    mkdir -p $3/collisions/pics/random_pose/
    GAZEBO_MODEL_PATH="$3/collisions/model/" GAZEBO_PLUGIN_PATH=$3/ModelPhotoShoot/build/:$GAZEBO_PLUGIN_PATH gzserver -s libmodelphotoshoot.so worlds/blank.world --propshop-save "$3/collisions/pics/random_pose/" --propshop-model "$3/collisions/model/$2" --data-file "$3/collisions/pics/random_pose/poses.txt" --random-joints

    cd ${_cur_dir}
    ./test_models.py "$3/collisions/model/" "$2" "$3/collisions/"
) }

_preprocess_sdf_and_materials() { (
    #convert .stl and .dae entries to .obj
    sed -i 's/.stl/.obj/g' "$1$2"
    sed -i 's/.dae/.obj/g' "$1$2"
    # Some sdfs have a comment before the xml tag
    # this makes the parser fail, since the tag is optional
    # we'll remove it as safety workaround
    sed -i '/<?xml*/d' "$1$2"

    find . -name '*.jpg' -type f -exec bash -c 'convert "$0" "${0%.jpg}.png"' {} \;
    find . -name '*.jpeg' -type f -exec bash -c 'convert "$0" "${0%.jpeg}.png"' {} \;

    find . -type f -name '*.mtl' -exec sed -i 's/.jpg/.png/g' '{}' \;
    find . -type f -name '*.mtl' -exec sed -i 's/.jpeg/.png/g' '{}' \;
) }

_provision_repos() { (
    set -eu
    cd ${_cur_dir}
    repo_dir=${PWD}/repos
    completion_token=2021-03-12.1
    completion_file=$1/.completion-token

    if [[ "$2" == *\.sdf ]]
    then
        _preprocess_sdf_and_materials "$1" "$2"
        ./render_ur_urdfs.py "$1" "$2"
    else
        if [[ -f ${completion_file} && "$(cat ${completion_file})" == "${completion_token}" ]]; then
        return 0
        fi
        set -x
        rm -rf ${repo_dir}

        mkdir ${repo_dir} && cd ${repo_dir}

        git clone https://github.com/ros-industrial/universal_robot
        cd universal_robot/
        git checkout e8234318cc94  # From melodic-devel-staging
        # Er... dunno what to do about this, so hackzzz
        cd ${_cur_dir}
        ./ros_setup.bash ./render_ur_urdfs.py "$1" "$2"
    fi

    echo "${completion_token}" > ${completion_file}
) }

_setup_venv() { (
    set -eu
    cd ${_cur_dir}
    completion_token="$(cat ./requirements.txt)"
    completion_file=${_venv_dir}/.completion-token

    if [[ -f ${completion_file} && "$(cat ${completion_file})" == "${completion_token}" ]]; then
        return 0
    fi

    set -x
    rm -rf ${_venv_dir}

    mkdir -p ${_venv_dir}
    tar -xzf $(_download_drake) -C ${_venv_dir} --strip-components=1

    # See: https://drake.mit.edu/python_bindings.html#inside-virtualenv
    python3 -m venv ${_venv_dir} --system-site-packages
    cd ${_venv_dir}
    ./bin/pip install -I pip wheel
    ./bin/pip install -I -r ${_cur_dir}/requirements.txt
    ./bin/pip freeze > ${_cur_dir}/requirements.freeze.txt

    echo "${completion_token}" > ${completion_file}
) }

if [  $# -lt "2" ]
    then
    echo "Please provide path to model directory and model file name."
    echo "      Usage:"
    echo "                  $./setup.sh <model_directory_path> <model_file_name> ./[executable]"
    echo "      or"
    echo "                  $source ./setup.sh <model_directory_path> <model_file_name>"
    echo "                  $./[executable]"

    exit 1
fi

temp_directory=$(mktemp -d)
echo "Saving temporal test files to: ${temp_directory}"

_build_gazebo_plugin "$temp_directory"

_setup_venv && source ${_venv_dir}/bin/activate

_provision_repos "$1" "$2"

_test_models "$1" "$2" "$temp_directory"

if [[ ${0} == ${BASH_SOURCE} ]]; then
    # This was executed, *not* sourced. Run arguments directly.
    set -eux
    #env
    exec "${@:3}"
fi
