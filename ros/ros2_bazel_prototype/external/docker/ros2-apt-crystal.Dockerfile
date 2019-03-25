# Based loosely on: https://hub.docker.com/r/osrf/ros2
# And: https://index.ros.org/doc/ros2/Installation/Linux-Install-Debians/
FROM ubuntu:bionic

# install packages
RUN apt-get update && apt-get install -q -y \
    gnupg2 \
    && rm -rf /var/lib/apt/lists/*

# setup ros2 keys
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 421C365BD9FF1F717815A3895523BAEEB01FA116

# setup sources.list
RUN echo "deb http://packages.ros.org/ros2/ubuntu bionic main" > /etc/apt/sources.list.d/ros2-latest.list

# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# setup timezone
RUN echo 'Etc/UTC' > /etc/timezone && \
    ln -s /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    apt-get update && apt-get install -q -y tzdata && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -q -y \
    ros-crystal-ros-base \
    && rm -rf /var/lib/apt/lists/*

# Install stuff for building.
RUN apt-get update && apt-get install -q -y \
    git \
    python3-colcon-common-extensions \
    python3-vcstool \
    && rm -rf /var/lib/apt/lists/*
