# -*- mode: python -*-
# vi: set ft=python :

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def libtorch_repository(
        name,
        **kwargs):
    http_archive(
        name = name,
        build_file = "//:libtorch.BUILD.bazel",
        strip_prefix = "libtorch",
        sha256 = "8d9e829ce9478db4f35bdb7943308cf02e8a2f58cf9bb10f742462c1d57bf287",  # noqa
        urls = ["https://download.pytorch.org/libtorch/cu113/libtorch-cxx11-abi-shared-with-deps-1.11.0%2Bcu113.zip"],  # noqa
        **kwargs
    )
