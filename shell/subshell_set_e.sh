#!/bin/bash

bash -ex <<'EOF'
false
echo "Should not reach"
EOF

bash -ex <<'EOF'
func() {
    false
    true
}

(
    func
)
echo "Should not reach"
EOF

set -eu
shopt -s expand_aliases

