#!/bin/bash
l
# Install minimal prereqs, except for MATLAB
# Adapted from: drake:51a89a6:setup/ubuntu/16.04/install_prereqs.sh

apt install --no-install-recommends $(echo "
    bash-completion
    default-jdk
    g++
    git
    make
    perl
    pkg-config
    python-dev
    python-numpy
    unzip
    wget
    zip
    zlib1g-dev
    ")

# Cannot build without clang??? Bad setup with repro/tools/CROSSTOOL or repro/tools/bazel.rc?
# Or is it a Drake thing?
apt-get install --no-install-recommends lsb-core software-properties-common wget
wget -q -O - http://llvm.org/apt/llvm-snapshot.gpg.key | sudo apt-key add -
add-apt-repository -y "deb http://apt.llvm.org/xenial/ llvm-toolchain-xenial-3.9 main"
apt-get update
apt install --no-install-recommends clang-3.9 lldb-3.9

# Install Bazel.
# has_bazel=$({ { dpkg -s bazel | grep 'Version: 0.4.5'; } > /dev/null 2>&1; } && echo 1)
wget -O /tmp/bazel_0.4.5-linux-x86_64.deb https://github.com/bazelbuild/bazel/releases/download/0.4.5/bazel_0.4.5-linux-x86_64.deb
if echo "b494d0a413e4703b6cd5312403bea4d92246d6425b3be68c9bfbeb8cc4db8a55 /tmp/bazel_0.4.5-linux-x86_64.deb" | sha256sum -c -; then
  dpkg -i /tmp/bazel_0.4.5-linux-x86_64.deb
else
  echo "The Bazel deb does not have the expected SHA256.  Not installing Bazel."
  exit 1
fi
rm /tmp/bazel_0.4.5-linux-x86_64.deb

# Setup quick and dirty environment variables
cat <<EOF >> ~/.bash_aliases
export PYTHONPATH=
export CC=gcc-5 CXX=g++-5 GFORTRAN=gfortran-5
EOF
