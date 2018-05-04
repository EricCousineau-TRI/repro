import os

from nbformat.v4 import nbjson as nbjson4
from nbformat.v3 import nbpy as nbpy3
from nbformat.converter import convert

# Load v4 notebook.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

src = "notebooks/test.ipynb"
dst = "notebooks/test.v3.py"

with open(src) as f:
    src_nb4 = nbjson4.read(f)

src_nb3 = convert(src_nb4, to_version=3)

with open(dst, 'w') as f:
    dst_nb = nbpy3.write(src_nb3, f)
