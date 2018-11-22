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
from os.path import basename
from subprocess import check_output
import sys

import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
args = parser.parse_args()

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

with open(args.output, 'w') as f:
    yaml.dump(data, f, default_flow_style=False)
