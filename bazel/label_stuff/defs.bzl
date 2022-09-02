def format_label(label_str):
    label = Label(label_str)
    return "{}:\n  workspace: {}\n  package: {}\n  name: {}".format(
        label,
        label.workspace_name,
        label.package,
        label.name,
    )

def test_stuff():
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
Bazel v5.1.0

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
