def clang_repository(name):
    native.new_local_repository(
        name = "clang",
        path = "/usr/lib/llvm-9",
        build_file = "@pybind_clang///tools/workspace/clang:package.BUILD.bazel",
    )
