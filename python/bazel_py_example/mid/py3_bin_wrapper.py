# N.B. This is executed in Python2, but then reroutes to Python3.
import os
import sys

args = ["python3", "mid/py3_bin.py"] + sys.argv[1:]
os.execvp(args[0], args)
