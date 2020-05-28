FROM ubuntu:bionic

WORKDIR /drake
COPY drake.tar.gz .
RUN tar -xvzf ./drake.tar.gz -C . --strip-components=1
RUN set -eux \
  && export DEBIAN_FRONTEND=noninteractive \
  && yes | ./share/drake/setup/install_prereqs \
  && rm -rf /var/lib/apt/lists/*
