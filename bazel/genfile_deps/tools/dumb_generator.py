"""Dumb example, but shows dependencies.

Example input file, `data/input.yaml`

    value: input
    include_file: data/input_extra.yaml

Example extra file, `data/input_extra.yaml`

    value_extra: input_extra

Generated output:

    value: input
    value_extra: input_extra

"""

import argparse
import os
from os.path import abspath, basename, dirname, isfile, join
from subprocess import check_output
import sys

import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--use_workaround", action="store_true")
args = parser.parse_args()

output = args.output

if args.use_workaround:
    me_relpath = "tools/dumb_generator.py"
    output = abspath(args.output)
    os.chdir(join(dirname(abspath(__file__)), '..'))
    assert isfile(me_relpath), "Meant for genfiles"

print([basename(__file__)] + sys.argv[1:])
files = check_output(["find", "."])
for line in files.split("\n"):
    print("    " + line)

with open(args.input) as f:
    data = yaml.load(f)

while True:
    include_file = data.pop("include_file", None)
    if include_file is None:
        break
    with open(include_file) as f:
        data.update(yaml.load(f))

with open(output, 'w') as f:
    yaml.dump(data, f, default_flow_style=False)
