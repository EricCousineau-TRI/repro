#!/bin/bash
set -eux

# Does not hang.
echo "[ Will not hang ]"
cat > /tmp/stuff.txt <<EOF
Hey stuff
EOF

echo "[ Will also not hang ]"
cat <<EOF | tee /tmp/stuff.txt
Hey stuff
EOF

# TODO(eric.cousineau): root cause this at some point?
echo "[ Will hang ]"
cat | tee /tmp/stuff.txt <<EOF
Hey stuff
EOF
