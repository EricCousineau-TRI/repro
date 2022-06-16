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
        sha256 = "2557943af80ec93f8249f6c5c829db6c6688842afa25a7d848f5c471473eb898",  # noqa
        urls = ["https://download.pytorch.org/libtorch/cu113/libtorch-cxx11-abi-shared-with-deps-1.10.2%2Bcu113.zip"],  # noqa
        **kwargs
    )
