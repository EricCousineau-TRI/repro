#!/bin/bash
set -eux -o pipefail

[[ "${EUID}" -eq 0 ]] || { echo "Must run as root."; exit 1; }

cat > /etc/udev/rules.d/90-footpedal.rules <<EOF
SUBSYSTEMS=="usb", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="e026", TAG+="uaccess"
SUBSYSTEMS=="usb", ATTRS{idVendor}=="3553", ATTRS{idProduct}=="b001", TAG+="uaccess"
EOF

# https://unix.stackexchange.com/questions/39370/how-to-reload-udev-rules-without-reboot
udevadm control --reload-rules
udevadm trigger
