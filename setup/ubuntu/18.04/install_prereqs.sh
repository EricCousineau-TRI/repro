#!/bin/bash

# Install minimal prereqs, except for MATLAB
# Adapted from: drake:51a89a6:setup/ubuntu/16.04/install_prereqs.sh

apt install --no-install-recommends $(echo "
    apt-transport-https
    bash-completion
    ca-certificates
    g++
    gdb
    git
    gnupg
    make
    pkg-config
    python
    python3-dev
    python3-numpy
    unzip
    wget
    zip
    zlib1g-dev
    ")

dpkg_install_from_wget() {
  package="$1"
  version="$2"
  url="$3"
  checksum="$4"

  # Skip the install if we're already at the exact version.
  installed=$(dpkg-query --showformat='${Version}\n' --show "${package}" 2>/dev/null || true)
  if [[ "${installed}" == "${version}" ]]; then
    echo "${package} is already at the desired version ${version}"
    return
  fi

  # If installing our desired version would be a downgrade, ask the user first.
  if dpkg --compare-versions "${installed}" gt "${version}"; then
    echo "This system has ${package} version ${installed} installed."
    echo "Drake suggests downgrading to version ${version}, our supported version."
    read -r -p 'Do you want to downgrade? [Y/n] ' reply
    if [[ ! "${reply}" =~ ^([yY][eE][sS]|[yY])*$ ]]; then
      echo "Skipping ${package} ${version} installation."
      return
    fi
  fi

  # Download and verify.
  tmpdeb="/tmp/${package}_${version}-amd64.deb"
  wget -O "${tmpdeb}" "${url}"
  if echo "${checksum} ${tmpdeb}" | sha256sum -c -; then
    echo  # Blank line between checkout output and dpkg output.
  else
    echo "The ${package} deb does not have the expected SHA256. Not installing." >&2
    exit 2
  fi

  # Install.
  dpkg -i "${tmpdeb}"
  rm "${tmpdeb}"
}

dpkg_install_from_wget \
  bazel 0.23.1 \
  https://github.com/bazelbuild/bazel/releases/download/0.23.1/bazel_0.23.1-linux-x86_64.deb \
  62d7fc733cb64c8bcedec4374e674ffacdc6616584d913fe84b97753c5e0863e
