#!/bin/bash
set -eux

bazel version

rm -rf build
mkdir build && cd build

# Create external repository.
mkdir repo && pushd repo
cat > WORKSPACE <<EOF
workspace(name = "repo")
EOF
cat > BUILD <<EOF
py_library(
    name = "repo_py",
    deps = ["//repo"],
    visibility = ["//visibility:public"],
)
EOF
# Simulate nested Python package with same name as parent.
mkdir repo && cd repo
cat > BUILD <<EOF
py_library(
    name = "repo",
    srcs = ["__init__.py"],
    imports = [".."],
    visibility = ["//:__pkg__"],
)
EOF
cat > __init__.py <<EOF
my_special_symbol = 1
EOF
popd

mkdir example && pushd example
cat > WORKSPACE <<EOF
workspace(name = "example")

local_repository(
    name = "repo",
    path = "../repo",
)
EOF
cat > BUILD <<EOF
py_binary(
    name = "example_consuming_repo",
    srcs = ["example_consuming_repo.py"],
    deps = ["@repo//:repo_py"],
)
EOF
cat > print_paths.py <<EOF

EOF
cat > example_consuming_repo.py <<EOF
# print paths
import os, re, sys

def reformat(x):
    source = os.path.dirname(os.path.realpath(__file__))
    x = x.replace(source, "{source}")
    x = re.sub(r"^/.*?\.runfiles", "{runfiles}", x)
    return x

print("\n".join(reformat(x) for x in sys.path))

# try import
from repo import my_special_symbol
EOF

echo "---"
bazel run -s //:example_consuming_repo
