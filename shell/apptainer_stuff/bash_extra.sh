_apptainer_stuff=$(cd $(dirname ${BASH_SOURCE}) && pwd)

apptainer-ros-jammy() {
    apptainer --silent exec \
        --nv --writable "$@" \
        ${_apptainer_stuff}/jammy-ros-humble-desktop.sif.sandbox \
        bash
}
