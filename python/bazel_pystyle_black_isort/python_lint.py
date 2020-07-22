"""
Check Python code for semantic errors (bad imports, etc.).
"""

import argparse
import json
from os.path import isfile
import re
import sys
from textwrap import indent

# N.B. black's public API is a bit unstable.
import black
import isort

KNOWN_PYTORCH_PACKAGES = [
    "torch",
    "torchvision",
]


def eprint(*args):
    print(*args, file=sys.stderr)


def get_import_package(line):
    """
    If this is an import, return the top-level package and indentation.
    Otherwise, return None.
    """
    prog = re.compile(r"(\s*)(import|from)\s+([\w\.]+)\b.*")
    m = prog.fullmatch(line)
    if m is None:
        return None, None
    else:
        prefix, _, module = m.groups()
        package = module.split(".")[0]
        return prefix, package


def check_maybe_preload_pydrake_for_torch(text, filename):
    messages = []
    imports_preload_pydrake_at = None
    imports_torch = False
    new_lines = []
    for i, line in enumerate(text.split("\n")):
        prefix, package = get_import_package(line)
        if package == "maybe_preload_pydrake_for_torch":
            imports_preload_pydrake_at = i
        elif package in KNOWN_PYTORCH_PACKAGES:
            if not imports_torch:
                imports_torch = True
                if imports_preload_pydrake_at is None:
                    new_line = (
                        f"{prefix}import maybe_preload_pydrake_for_torch")
                    new_lines.append(new_line)
                    messages.append(
                        f"{filename}:{i + 1}: Please call "
                        f"`{new_line.strip()}` before importing "
                        f"`{package}`.")
        new_lines.append(line)
    if not imports_torch and imports_preload_pydrake_at is not None:
        messages.append(
            f"{filename}:{imports_preload_pydrake_at + 1}: torch is not used, "
            f"please remove this line")
        del new_lines[imports_preload_pydrake_at]
    return messages, "\n".join(new_lines)


def check_file(text, filename, use_black=False, isort_settings_file=None):
    if filename.endswith(".ipynb"):
        messages = []
        doc = json.loads(text)
        assert doc["metadata"]["kernelspec"]["language"] == "python"

        for i, cell in enumerate(doc["cells"]):
            cell_name = f"{filename}/In[{i + 1}]"
            if cell["cell_type"] == "code":
                cell_text = "".join(cell["source"])
                try:
                    cell_messages, new_cell_text = check_file(
                        cell_text,
                        cell_name,
                        use_black=use_black,
                        isort_settings_file=isort_settings_file,
                    )
                except black.InvalidInput:
                    cell_messages = []
                    new_cell_text = cell_text
                    messages.append(f"{cell_name}: Could not parse!")
                cell["source"] = new_cell_text.rstrip().splitlines(keepends=True)
                messages += cell_messages
                if len(cell["outputs"]) > 0:
                    messages.append(f"{cell_name}: There should be no outputs!")
                    cell["outputs"] = []

        new_text = json.dumps(doc, indent=1) + "\n"

        return messages, new_text

    messages, new_text = check_maybe_preload_pydrake_for_torch(
        text, filename)
    new_text_orig = new_text

    formatters_used = []
    if use_black:
        formatters_used += ["black"]
        black_file_mode = black.FileMode(
            target_versions={black.TargetVersion.PY36},
            line_length=79,
        )
        # WARNING: This may create line-length problems:
        # https://github.com/psf/black/issues/208
        # https://github.com/psf/black/issues/1017
        try:
            new_text = black.format_file_contents(
                src_contents=new_text,
                fast=True,
                mode=black_file_mode,
            )
        except black.NothingChanged:
            pass

    if isort_settings_file is not None:
        # N.B. If isort_settings_file == "", then it will load a default file.
        if isort_settings_file != "":
            assert isfile(isort_settings_file), isort_settings_file
        formatters_used += ["isort"]
        isort_config = isort.Config(settings_file=isort_settings_file)
        new_text = isort.code(new_text, config=isort_config)

    # N.B. Check after applying both, as they may conflict between each other
    # and we don't care about intermediate results.
    if new_text != new_text_orig:
        messages.append(
            f"{filename}: Needs reformatting for {formatters_used}")

    return messages, new_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix", action="store_true")
    parser.add_argument("--use_black", action="store_true")
    parser.add_argument("--isort_settings_file", type=str, default=None)
    parser.add_argument("files", type=str, nargs='*')
    args = parser.parse_args()

    fix_args = ["--fix"]
    if args.use_black:
        fix_args += [f"--use_black"]
    if args.isort_settings_file is not None:
        fix_args += [f"--isort_settings_file={args.isort_settings_file}"]
    fix_args_str = " ".join(fix_args)

    messages = []
    for file in sorted(args.files):
        with open(file, "r", encoding="utf8") as f:
            text = f.read()
        new_messages, new_text = check_file(
            text,
            file,
            use_black=args.use_black,
            isort_settings_file=args.isort_settings_file,
        )
        if new_text != text:
            if args.fix:
                with open(file, 'w', encoding="utf8") as f:
                    f.write(new_text)
            else:
                messages += new_messages
                messages.append(
                    f"To fix, run:\n"
                    f"  bazel-bin/tools/lint/python_lint {fix_args_str} "
                    f"{file}\n")

    if messages:
        messages_str = "\n".join(messages)
        eprint()
        eprint(messages_str)
        eprint("You may need to build the tools first before using --fix:")
        eprint("  bazel build //tools/lint/...")
        eprint()
        if "maybe_preload_pydrake_for_torch" in messages_str:
            eprint("WARNING: If you need maybe_preload_pydrake_for_torch, be ")
            eprint("sure to make the necessary bazel targets depend on ")
            eprint("\"//tools:pytorch\", not just \"@pytorch\".")
            eprint()
        sys.exit(1)


if __name__ == "__main__":
    main()
