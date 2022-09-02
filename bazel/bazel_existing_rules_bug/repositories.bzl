def _make_a(repo_ctx):
    repo_ctx.file(
        "BUILD.bazel",
        """
package(default_visibility = ["//visibility:public"])
load("@//:analysis.bzl", "print_rule_deps")

cc_library(
    name = "a_lib",
)

print_rule_deps()
        """.rstrip(),
    )

make_a = repository_rule(
    implementation = _make_a,
)

def _make_b(repo_ctx):
    repo_ctx.file(
        "BUILD.bazel",
        """
package(default_visibility = ["//visibility:public"])
load("@//:analysis.bzl", "print_rule_deps")

cc_library(
    name = "b_lib",
    deps = ["@a//:a_lib"],
)

print_rule_deps()
    """.rstrip(),
    )

make_b = repository_rule(
    implementation = _make_b,
)
