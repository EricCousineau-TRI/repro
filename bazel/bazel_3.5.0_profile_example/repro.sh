#!/bin/bash

# To run:
#
#   ./repro.sh

set -eux -o pipefail

main() { (
uname -a
cat /etc/issue

cd $(mktemp -d)

cat > WORKSPACE <<EOF
workspace(name = "test")
EOF

cat > BUILD.bazel <<EOF
sh_test(
    name = "sample_test",
    srcs = ["sample_test.sh"],
)
EOF

cat > sample_test.sh <<EOF
#!/bin/bash

echo "I am a test. kthxbye"
EOF
chmod +x ./sample_test.sh

bazel version
bazel build ...
bazel test --nocache_test_results --profile=./profile.json ...

bazel analyze-profile ./profile.json
bazel analyze-profile --dump=raw ./profile.json

# Errors...
set +e
bazel analyze-profile --html ./profile.json
bazel help --long analyze-profile | grep html
) }

cd $(dirname ${BASH_SOURCE})
main 2>&1 | tee ./repro.output.txt
