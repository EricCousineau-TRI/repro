#!/usr/bin/env python

from memory_profiler import profile
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-a', '--all_the_things', type=str, default='Default')

@profile
def main():
    args = parser.parse_args()
    print args.all_the_things

main()
