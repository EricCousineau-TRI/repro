# Modified from:
# drake-distro:3aec5a9:setup/docker/Dockerfile.ubuntu16.04.opensource

FROM ubuntu:18.04
COPY . /root/repro
RUN apt update && \
     yes "Y" | /root/repro/setup/ubuntu/18.04/install_prereqs.sh \
     && rm -rf /var/lib/apt/lists/* \
     && apt-get clean all
RUN cd /root/repro && bazel run //cpp/eigen:main
ENTRYPOINT ["bash"]
