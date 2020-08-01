def pybind11_repository(name):
    native.new_local_repository(
        name = "pybind11",
        path = "/home/eacousineau/proj/tri/repo/externals/pybind11",
        build_file = "@pybind_clang//tools/workspace/pybind11:package.BUILD.bazel",
    )
