#!/usr/bin/env python

import argparse
import os

from nbformat import v4
from nbformat import v3
from nbformat.converter import convert


def v4_json_to_v3_py(fin, fout):
    nbin_v4 = v4.nbjson.read(fin)
    nbin_v3 = convert(nbin_v4, to_version=3)
    v3.nbpy.write(nbin_v3, fout)


def v3_py_to_v4_json(fin, fout):
    nbin_v3 = v3.nbpy.read(fin)
    nbin_v4 = convert(nbin_v3, to_version=4)
    v4.nbjson.write(nbin_v4, fout)



parser = argparse.ArgumentParser()
parser.add_argument("--to_python", action="store_true")
parser.add_argument("--from_python", action="store_true")
parser.add_argument("input", type=argparse.FileType('r'))
parser.add_argument("output", type=argparse.FileType('w'))

args = parser.parse_args()

assert args.to_python ^ args.from_python, "Must choose one format"

func = None
if args.to_python:
    func = v4_json_to_v3_py
elif args.from_python:
    func = v3_py_to_v4_json

assert func is not None

func(args.input, args.output)
