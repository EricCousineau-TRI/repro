def format_label(label_str):
    label = Label(label_str)
    return "{}:\n  workspace: {}\n  package: {}\n  name: {}".format(
        label,
        label.workspace_name,
        label.package,
        label.name,
    )

def test_labels():
    lines = [
        "",
        format_label("//:lib"),
        format_label("//pkg"),
        format_label("//pkg:lib"),
        format_label("@repo//:lib"),
        format_label("@repo//pkg"),
        format_label("@repo//pkg:lib"),
    ]
    print("\n".join(lines))

"""
$ bazel build //...
...
//:lib:
  workspace:
  package:
  name: lib
//pkg:pkg:
  workspace:
  package: pkg
  name: pkg
//pkg:lib:
  workspace:
  package: pkg
  name: lib
@repo//:lib:
  workspace: repo
  package:
  name: lib
@repo//pkg:pkg:
  workspace: repo
  package: pkg
  name: pkg
@repo//pkg:lib:
  workspace: repo
  package: pkg
  name: lib
"""

def _make_repo(repo_ctx):
    repo_ctx.file(
        "BUILD.bazel",
        """
load("@//:defs.bzl", "test_scopes")
test_scopes("In @repo")
        """.rstrip(),
    )
    repo_ctx.file(
        "pkg/BUILD.bazel",
        """
load("@//:defs.bzl", "test_scopes")
test_scopes("In @repo//pkg")
""")

make_repo = repository_rule(
    implementation = _make_repo,
)

def test_scopes(note):
    print("\n{}\n  repository_name: {}\n  package_name: {}".format(
        note,
        native.repository_name(),
        native.package_name(),
    ))

"""
$ bazel build //... @repo//... 2>&1 | sed -E 's#^DEBUG.*$##g'
...
In //pkg
  repository_name: @
  package_name: pkg

In //
  repository_name: @
  package_name:

In @repo//pkg
  repository_name: @repo
  package_name: pkg

In @repo
  repository_name: @repo
  package_name:
"""
