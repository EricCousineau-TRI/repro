#!/usr/bin/env python

import os

print("File:")
with open('bazel/input.txt') as f:
    print(f.read())

print("Writing")
with open('bazel/input.txt', 'a') as f:
    f.write("Write new line")
