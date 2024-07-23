#!/bin/bash
set -eux -o pipefail

# Keep corefiles + backtraces around for a while.
# Users run "coredumpctl" to manipulate the cores database.
sudo apt-get install systemd-coredump

# Obey coredump size rlimits (systemd ignores them by default).
sudo tee /etc/sysctl.d/50-coredump.conf <<EOF
# Added by coredump setup.
kernel.core_pattern=|/lib/systemd/systemd-coredump %P %u %g %s %t %c %e
EOF
sudo sysctl -p /etc/sysctl.d/50-coredump.conf

# Raise the core size limits up from 2G.
mkdir -p /etc/systemd/coredump.conf.d
sudo tee /etc/systemd/coredump.conf.d/50-enabling_anzu.conf <<EOF
# Added by coredump setup.
[Coredump]
ProcessSizeMax=4G
ExternalSizeMax=16G
EOF
