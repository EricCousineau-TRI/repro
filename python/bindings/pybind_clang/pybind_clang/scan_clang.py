import os
from os.path import dirname
from subprocess import run, PIPE
# :(
from autopybind11.__main__ import BindingsGenerator, parse_options

# :( :(
fake_response = """
includes: /usr/lib/llvm-9/include
defines:
c_std: --std=c++17
"""


def main():
    os.chdir(dirname(__file__))

    output_dir = "/tmp/autopybind11"
    rsp_path = f"{output_dir}/fake_response.txt"
    with open(rsp_path, "w") as f:
        f.write(fake_response)

    castxml_path = run(
        "which castxml",
        shell=True, encoding="utf8", stdout=PIPE, check=True
    ).strip()

    argv = [
        "--output", f"{output_dir}/output",
        "--input_yaml", "scan_clang_config.yml",
        "--castxml-path", castxml_path,
        "--input_response", rsp_path,        
    ]

    options = parse_options(argv=argv)
    gen = BindingsGenerator(options)
    gen.parse_and_generate()


if __name__ == "__main__":
    main()
