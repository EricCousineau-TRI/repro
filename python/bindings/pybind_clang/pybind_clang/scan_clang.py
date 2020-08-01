import os
from os.path import dirname, join
from subprocess import run, PIPE
from textwrap import dedent

# :(
from autopybind11.__main__ import BindingsGenerator, parse_options


def main():
    os.chdir(dirname(__file__))

    llvm_inc = "/usr/lib/llvm-9/include"

    tmp_dir = "/tmp/autopybind11"
    input_dir = join(tmp_dir, "input")
    output_dir = join(tmp_dir, "output")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # :( :( :(
    fake_response = dedent(f"""\
    includes: {llvm_inc};{output_dir}
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
        "--input_yaml", "./scan_clang_config.yml",
        "--module_name", "sample_auto",
        "--castxml-path", castxml_path,
        "--input_response", rsp_path,        
    ]

    options = parse_options(argv=argv)
    gen = BindingsGenerator(options)
    gen.parse_and_generate()


if __name__ == "__main__":
    main()
