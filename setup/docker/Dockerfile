# Modified from:
# drake-distro:3aec5a9:setup/docker/Dockerfile.ubuntu16.04.opensource
FROM ubuntu:16.04
COPY . ~/repro
RUN yes "Y" \
      | ~/repro/setup/ubuntu/16.04/install_prereqs.sh \
      && rm -rf /var/lib/apt/lists/* \
      && apt-get clean all
RUN cd ~/repro && bazel run //python/bindings/pymodule:typebinding_test
ENTRYPOINT ["bash"]
