#!/bin/bash
#
# Prerequisite set-up script for my_proj.

set -euo pipefail

die () {
    echo "$@" 1>&2
    trap : EXIT  # Disable line number reporting; the "$@" message is enough.
    exit 1
}

at_exit () {
    echo "${me} has experienced an error on line ${LINENO}" \
        "while running the command ${BASH_COMMAND}"
}

cat_without_comments() {
    sed -e 's|#.*||g;' "$(dirname $0)/${ubuntu_release_yymm}/$1"
}

# Run `apt-get install` on the arguments, then clean up.
apt_install () {
  sudo apt-get -q install --no-install-recommends "$@"
  sudo apt-get -q clean
}

deb_install() {
  local url=${1}
  local sha=${2}
  local base=$(basename ${1})
  wget ${url} -O /tmp/${base}
  echo "${sha} /tmp/${base}" | sha256sum -c -
  apt_install /tmp/${base}
}

me="The my_proj prerequisite set-up script"

trap at_exit EXIT

# Check that script wasn't run with sudo.
if [[ "${EUID}" -eq 0 ]]; then
  die "${me} must NOT be run as root."
fi

# Verify sudo. Do this separately so there's a useful error message if the user
# gets their password wrong. Don't run this check in Jenkins or if explicitly
# disabled.
if [[ ! -v HUDSON_HOME && ! -v _NO_SUDO_VERIFY ]]; then
  sudo -v
else
  echo "Not verifying sudo."
fi

# Check for supported versions by installing and then using lsb_release.
sudo apt-get update -y || true
apt_install lsb-release apt-transport-https ca-certificates gnupg wget
ubuntu_release_yymm=$(lsb_release -sr)
ubuntu_codename=$(lsb_release -sc)
if [[ \
    "${ubuntu_release_yymm}" != "22.04" ]]; then
  die "${me} only supports Ubuntu 22.04."
fi

# Install APT dependencies.
# (We do them piecewise to minimize our download footprint.)
apt_install $(cat_without_comments packages-python.txt)

# Install bazelisk.
deb_install \
    https://github.com/bazelbuild/bazelisk/releases/download/v1.22.0/bazelisk-amd64.deb \
    f3a9dd15b08f3f1350f2b2055cfee8a9c412c2050966f635633aaf30dd7e979e

# Bootstrap the venv. The `setup` is required here (nothing else runs it), but
# the `sync` is not strictly requied (bazel will run it for us). We do it here
# anyway so that the user gets immediate, streaming feedback about problems
# with internet downloads, or other kinds of glitches. (When run from Bazel,
# stderr text is batched up until much later, not streamed.)
$(dirname $0)/../tools/workspace/venv/setup
$(dirname $0)/../tools/workspace/venv/sync

trap : EXIT  # Disable exit reporting.
echo "install_prereqs: success"
