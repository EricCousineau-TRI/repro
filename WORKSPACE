workspace(name = "repro")

load("//tools:github.bzl", "github_archive")
load("//tools:bitbucket.bzl", "bitbucket_archive")

bitbucket_archive(
    name = "eigen",
    repository = "eigen/eigen",
    commit = "3.3.3",
    sha256 = "94878cbfa27b0d0fbc64c00d4aafa137f678d5315ae62ba4aecddbd4269ae75f",  # noqa
    strip_prefix = "eigen-eigen-67e894c6cd8f",
    build_file = "@repro//tools:eigen.BUILD.bazel",
)

load("//tools:python.bzl", "python_repository")
python_repository(
    name = "python",
    version = "2.7",
)

python_repository(
    name = "python3",
    version = "3.5",
)

load("//tools:numpy.bzl", "numpy_repository")
numpy_repository(
    name = "numpy",
    python_version = "2.7",
)

new_local_repository(
    name = "pybind11",
    path = "externals/pybind11",
    build_file = "@repro//tools:pybind11.BUILD.bazel",
)

github_archive(
    name = "gtest",
    repository = "google/googletest",
    commit = "release-1.8.0",
    sha256 = "58a6f4277ca2bc8565222b3bbd58a177609e9c488e8a72649359ba51450db7d8",  # noqa
    build_file = "@repro//tools:gtest.BUILD.bazel",
)

# Hack in VTK
load("//tools:vtk.bzl", "vtk_repository")
vtk_repository(
    name = "vtk",
)

github_archive(
    name = "fmt",
    repository = "fmtlib/fmt",
    commit = "4.1.0",
    sha256 = "46628a2f068d0e33c716be0ed9dcae4370242df135aed663a180b9fd8e36733d",  # noqa
    build_file = "@repro//tools:fmt.BUILD.bazel",
)
