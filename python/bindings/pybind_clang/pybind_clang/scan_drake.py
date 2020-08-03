import os
from os.path import dirname, join, expanduser
from subprocess import run, PIPE
from textwrap import dedent

# :(
from autopybind11.__main__ import BindingsGenerator, parse_options


def main():
    os.chdir(dirname(__file__))

    drake_path = expanduser("~/venv/drake")

    tmp_dir = "/tmp/autopybind11_drake"
    input_dir = join(tmp_dir, "input")
    output_dir = join(tmp_dir, "output")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # :( :( :(
    fake_response = dedent(f"""\
    includes: {output_dir};{drake_path}/include;{drake_path}/include/fmt;/usr/include/eigen3
    defines:
    c_std: --std=c++17
    """).strip()

    rsp_path = join(input_dir, "fake_response.txt")
    with open(rsp_path, "w") as f:
        f.write(fake_response)

    castxml_path = run(
        "which castxml",
        shell=True, encoding="utf8", stdout=PIPE, check=True
    ).stdout.strip()

    argv = [
        "--output", output_dir,
        "--input_yaml", "./scan_drake_config.yml",
        "--module_name", "pydrake",
        "--castxml-path", castxml_path,
        "--input_response", rsp_path,
        "--start_with_declarations", "drake",
    ]

    options = parse_options(argv=argv)
    gen = BindingsGenerator(options)
    gen.parse_and_generate()


if __name__ == "__main__":
    main()
