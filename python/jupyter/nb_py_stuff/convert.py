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

parser = argparse.ArgumentParser()
parser.add_argument("--to", type=str, choices=choices, default="json")
parser.add_argument("--from", dest="from_", type=str, choices=choices, default="py_v3")
parser.add_argument("input", type=argparse.FileType('r'))
parser.add_argument("output", type=argparse.FileType('w'))

args = parser.parse_args()

(v_from, m_from) = conversions[args.from_]
(v_out, m_to) = conversions[args.to]

nb_from = m_from.read(args.input)
if v_from != v_out:
    nb_from = convert(nb_from, to_version=v_out)
m_to.write(nb_from, args.output)
