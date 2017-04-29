workspace(name = "repro")

load("//tools:github.bzl", "github_archive")
load('@bazel_tools//tools/build_defs/repo:git.bzl', 'git_repository')

github_archive(
    name = "eigen",
    repository = "RobotLocomotion/eigen-mirror",
    commit = "d3ee2bc648be3d8be8c596a9a0aefef656ff8637",
    build_file = "tools/eigen.BUILD",
    sha256 = "db797e2857d3d6def92ec2c46aa04577d3e1bb371d6fe14e6bdfc088dcaf2e9e",
    local_repository_override = "externals/eigen",  # This is present ahead of the listed commit by 10 or so commits
)
