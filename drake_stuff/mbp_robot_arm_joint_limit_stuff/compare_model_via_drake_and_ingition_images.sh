#!/bin/bash

_cur_dir=$(cd $(dirname ${BASH_SOURCE}) && pwd)

_test_models() { (

    cd "$3"
    mkdir -p visual/model/
    cp -r "$1"/* visual/model/
    cd visual/model/
    sed -i 's/'"'"'/"/g' "$2"
    MODEL_NAME=`grep 'model name' model.sdf|cut -f2 -d '"'`
    sed -i "s,model://$MODEL_NAME/,,g" "$2"

    # Generate model pics using the gazebo plugin.
    mkdir -p $3/visual/pics/default_pose/
    cd "$3/visual/pics/default_pose/"
    python -c "import sdf_manipulation; sdf_manipulation.generate_sdf('$3/visual/model/$2', '$3/visual/pics/default_pose/poses.txt', 'false', '$3/visual/pics/default_pose/plugin_config.sdf')"
    ign gazebo -s -r "$3/visual/pics/default_pose/plugin_config.sdf" --iterations 50

    mkdir -p $3/visual/pics/random_pose/
    cd "$3/visual/pics/random_pose/"
    python -c "import sdf_manipulation; sdf_manipulation.generate_sdf('$3/visual/model/$2', '$3/visual/pics/random_pose/poses.txt', 'true', '$3/visual/pics/random_pose/plugin_config.sdf')"
    ign gazebo -s -r "$3/visual/pics/random_pose/plugin_config.sdf" --iterations 50

    # Generate model pics using drake then run
    # IoU tests and extra checks.
    cd ${_cur_dir}
    ./test_models.py "$1" "$2" "$3/visual/"

    # Test collision meshes:
    # Exchange visual and collision meshes in the model
    # in order to take pics of the collision mesh and
    # compare them.
    cd "$3"
    mkdir -p collisions/model/
    cp -r "$1"/* collisions/model/
    cd collisions/model/
    find . -name "*.mtl" | xargs rm
    sed -i 's/collision/temporalname/g' "$2"
    # Workaround to ignore the old collision tags
    sed -i 's/visual/ignore:collision/g' "$2"
    sed -i 's/temporalname/visual/g' "$2"

    sed -i 's/'"'"'/"/g' "$2"
    sed -i "s,model://$MODEL_NAME/,,g" "$2"

    # Generate model pics using the gazebo plugin.
    mkdir -p $3/collisions/pics/default_pose/
    cd "$3/collisions/pics/default_pose/"
    python -c "import sdf_manipulation; sdf_manipulation.generate_sdf('$3/collisions/model/$2', '$3/collisions/pics/default_pose/poses.txt', 'false', '$3/collisions/pics/default_pose/plugin_config.sdf')"
    ign gazebo -s -r "$3/collisions/pics/default_pose/plugin_config.sdf" --iterations 50

    mkdir -p $3/collisions/pics/random_pose/
    cd "$3/collisions/pics/random_pose/"
    python -c "import sdf_manipulation; sdf_manipulation.generate_sdf('$3/collisions/model/$2', '$3/collisions/pics/random_pose/poses.txt', 'true', '$3/collisions/pics/random_pose/plugin_config.sdf')"
    ign gazebo -s -r "$3/collisions/pics/random_pose/plugin_config.sdf" --iterations 50

    # Generate model pics using drake then run
    # IoU tests and extra checks.
    cd ${_cur_dir}
    ./test_models.py "$3/collisions/model/" "$2" "$3/collisions/"
) }

if [[ $# -lt "2" ]]; then
    echo "Please provide path to model directory and model file name."
    echo "      Usage:"
    echo "                  $bash compare_model_via_drake_and_ignition_images.sh <model_directory_path> <model_file_name> "
    return 1
fi

temp_directory=$(mktemp -d)
echo "Saving temporal test files to: ${temp_directory}"

_test_models "$1" "$2" "$temp_directory"
