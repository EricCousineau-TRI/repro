# -*- python -*-

load("//tools/skylark:py.bzl", "py_test")

def python_lint_direct(
        name_prefix,
        files,
        use_black,
        isort_settings_file,
        tags = []):
    # Anzu Drake lint.
    args = ["$(location {})".format(f) for f in files]
    data = files
    tool = "//tools/lint:python_lint"
    tool_main = "//tools/lint:python_lint.py"
    if isort_settings_file != None:
        data = data + [isort_settings_file]
        args = [
            "--isort_settings_file=$(location {})".format(isort_settings_file),
        ] + args
    if use_black:
        args = ["--use_black"] + args
    py_test(
        name = name_prefix + "_python_lint",
        size = "small",
        srcs = [tool_main],
        deps = [tool],
        data = data,
        args = args,
        main = tool_main,
        tags = tags + ["python_lint", "lint"],
    )

def python_lint(
        existing_rules = None,
        exclude = [],
        extra_srcs = [],
        use_black = False,
        isort_settings_file = None):
    if existing_rules == None:
        existing_rules = native.existing_rules().values()
    for rule in existing_rules:
        # Disable linting when requested (e.g., for generated code).
        if "nolint" in rule.get("tags"):
            continue

        # Extract the list of python sources.
        srcs = rule.get("srcs", ())
        if type(srcs) == type(()):
            files = [
                s
                for s in srcs
                if s.endswith(".py") and s not in (exclude or [])
            ]
        else:
            # The select() syntax returns an object we (apparently) can't
            # inspect.  TODO(jwnimmer-tri) Figure out how to lint these files.
            files = []

        # Add a lint test if necessary.
        if files:
            python_lint_direct(
                name_prefix = rule["name"],
                files = files,
                use_black = use_black,
                isort_settings_file = isort_settings_file,
            )
    if extra_srcs:
        python_lint_direct(
            name_prefix = "extra_srcs",
            files = extra_srcs,
            use_black = use_black,
            isort_settings_file = isort_settings_file,
        )
