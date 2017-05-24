#!/bin/bash

# Helper functions for using docker, sharing X11
# @WARNING Be wary of security!

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

# @ref http://stackoverflow.com/questions/28302178/how-can-i-add-a-volume-to-an-existing-docker-container
# Caveat: Would be invalidated at startup?
# Can always commit the image if needing to change environment variable, mounting, etc.
# http://stackoverflow.com/questions/19158810/docker-mount-volumes-as-readonly
docker-x11-env() {
    XSOCK=/tmp/.X11-unix
    XAUTH=/tmp/.docker.xauth
    xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -
}

# Run an image, creating a container.
docker-x11-run() { (
  set -e -u -x
  container=${1}
  image=${2-ubuntu_x11}
  x11-docker-env
  docker run -ti --name ${container} -h ${container} -v $XSOCK:$XSOCK:ro -v $XAUTH:$XAUTH:ro -e XAUTHORITY=$XAUTH -e DISPLAY ${image}
) }

# (Re)start the container with added user (change CMD if possible)
docker-x11-exec() { (
    set -e -u -x
    # In docker root, expose XAuthority to user (cannot change read permission due to read-only mount)
    cmd_default='cp ${XAUTHORITY}{,.user} && chown user:user ${XAUTHORITY}.user; export XAUTHORITY=${XAUTHORITY}.user; su --login user'
    container=${1}
    cmd="${2-${cmd_default}}"
    docker-x11-env
    docker start ${container}
    docker exec -it -e XAUTHORITY=$XAUTH -e DISPLAY ${container} bash -c "${cmd}"
) }
