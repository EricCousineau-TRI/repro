#!/bin/bash
set -eux

# Adapted from Anzu.

# Check for root.
[[ "${EUID}" -eq 0 ]] || die "This must run as root. Please use sudo."

# String to insert into limits.conf to allow use of the realtime scheduler.
RT_PRIO_LIMIT='*                -       rtprio         90'
if ! grep -qF "$RT_PRIO_LIMIT" /etc/security/limits.conf; then
    echo "Enabling realtime scheduling."
    echo "$RT_PRIO_LIMIT" >> /etc/security/limits.conf
    echo "You may need to reboot for this change to take effect."
fi
