#!/bin/bash

# Helper functions for using docker, sharing X11
# @WARNING Be wary of security!

# @ref http://wiki.ros.org/docker/Tutorials/GUI
# @ref https://hub.docker.com/r/gtarobotics/udacity-sdc/

# Prequisite:
# * Docker image with Ubuntu
# * A user "user" with passwordless sudo
# * The necessary X11 packages installed

# Commands:
<<DOCKER_COMMANDS
    # From 'ubuntu' docker image

    apt install sudo x11-apps iputils-ping

    adduser --disabled-password --gecos '' user
    adduser user sudo

    cat <<EOF > /etc/sudoers.d/99-docker-user
    user ALL = NOPASSWD: ALL
    EOF
DOCKER_COMMANDS

# export DOCKER=docker
export DOCKER=nvidia-docker
DOCKER_SHARE_HOST=~/mnt/docker_share
DOCKER_SHARE_IMAGE=/mnt/docker_share

docker-x11-nvidia() {
  DOCKER=nvidia-docker
}

# @ref http://stackoverflow.com/questions/28302178/how-can-i-add-a-volume-to-an-existing-docker-container
# Caveat: Would be invalidated at startup?
# Can always commit the image if needing to change environment variable, mounting, etc.
# http://stackoverflow.com/questions/19158810/docker-mount-volumes-as-readonly
docker-x11-env() {
    XSOCK=/tmp/.X11-unix
    XAUTH=/tmp/.docker.xauth
    DOCKER_ENV_ARGS="-e XAUTHORITY=$XAUTH -e DISPLAY -e QT_X11_NO_MITSHM=1"
    xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -
}

# Run an image, creating a container.
docker-x11-run() { (
  set -e -u -x
  flags=
  while [[ $# -gt 0 ]]; do
    case $1 in
      --rm)
        flags="$flags --rm"
        shift;;
      *)
        break;;
    esac
  done
  image=${1}  # ubuntu_x11
  container=${2}
  cmd=${3-bash}
  docker-x11-env
  ${DOCKER} run -ti $flags --name ${container} -h ${container} \
      -v $XSOCK:$XSOCK:rw -v $XAUTH:$XAUTH:rw -v /dev/video0:/dev/video0 \
      -v $DOCKER_SHARE_HOST:$DOCKER_SHARE_IMAGE:shared \
      ${DOCKER_ENV_ARGS} \
      ${image}
) }

# (Re)start the container with added user (change CMD if possible)
docker-x11-exec() { (
    set -e -u -x
    no_start=
    while [[ $# -gt 0 ]]; do
      case $1 in 
        --no-start)
          no_start=1
          shift;;
        *)
          break;;
      esac
    done
    # In docker root, expose XAuthority to user (cannot change read permission due to read-only mount)
    # cmd_default='cp ${XAUTHORITY}{,.user} && chown user:user ${XAUTHORITY}.user; export XAUTHORITY=${XAUTHORITY}.user; su --login user'
    cmd_default='bash' # su --login user
    container=${1}
    cmd="${2-${cmd_default}}"
    docker-x11-env
    [[ -z $no_start ]] && ${DOCKER} start ${container}
    ${DOCKER} exec -it ${DOCKER_ENV_ARGS} ${container} "${cmd}"
) }
