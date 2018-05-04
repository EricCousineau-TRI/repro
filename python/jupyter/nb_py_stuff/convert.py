#!/usr/bin/env python

import argparse
import os

from nbformat import v4
from nbformat import v3
from nbformat.converter import convert

conversions = {
    "json": (4, v4.nbjson),
    "py_v3": (3, v3.nbpy),
}
choices = conversions.keys()

# https://stackoverflow.com/a/35720002/7829525
def pad_markdown(text):
    return text + "\n# <markdowncell>\n\n# Bugfix\n"

adjust_from_map = {
    ("py_v3", "json"): pad_markdown,
}

parser = argparse.ArgumentParser()
parser.add_argument("--to", type=str, choices=choices, default="json")
parser.add_argument("--from", dest="from_", type=str, choices=choices, default="py_v3")
parser.add_argument("input", type=argparse.FileType('r'))
parser.add_argument("output", type=argparse.FileType('w'))

args = parser.parse_args()

(v_from, m_from) = conversions[args.from_]
(v_out, m_to) = conversions[args.to]
adjust_from = adjust_from_map.get((args.from_, args.to))

text_from = args.input.read()
if adjust_from is not None:
    text_from = adjust_from(text_from)
nb_from = m_from.reads(text_from)
if v_from != v_out:
    nb_from = convert(nb_from, to_version=v_out)
m_to.write(nb_from, args.output)
