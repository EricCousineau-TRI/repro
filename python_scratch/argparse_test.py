#!/usr/bin/env python

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-a', '--all_the_things', type=str, default='Default')

args = parser.parse_args()
print args.all_the_things
