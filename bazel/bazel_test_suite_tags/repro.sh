#!/bin/bash
set -eu

bazel-test() {
    # filters="$1"
    # shift
    # targets="$@"

    echo "+ bazel test $@"
    bazel test "$@" \
        2>&1 \
        | grep '//.*FAILED' \
        | sed -E 's# +# #g' \
        | sed -E 's# in .*?s$##g'
    echo
}

bazel-test //tests/...
bazel-test //tests/... --test_tag_filters=a
bazel-test //tests/... --test_tag_filters=-b

bazel-test //suites:no_tags
bazel-test //suites:no_tags --test_tag_filters=a
bazel-test //suites:no_tags --test_tag_filters=-b

bazel-test //suites:a_tag
bazel-test //suites:minus_b_tag
