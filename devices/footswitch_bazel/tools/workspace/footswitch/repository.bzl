# -*- mode: python -*-
# vi: set ft=python :

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def footswitch_repository(
        name,
        mirrors = None):
    commit = "b7493170ecc956ac87df2c36183253c945be2dcf"
    url = (
        "https://github.com/rgerganov/footswitch/archive/{}.tar.gz"
        .format(commit)
    )
    sha256 = "40f18846241d4e1809778163ca1ca12f1bab20a1f74304f84f8bd33d21b480ac"
    http_archive(
        name = name,
        url = url,
        sha256 = sha256,
        strip_prefix = "footswitch-{}".format(commit),
        build_file = "//tools/workspace/footswitch:package.BUILD.bazel",  # noqa
    )
