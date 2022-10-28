"""
Grah!
https://stackoverflow.com/questions/7820809/understanding-weird-boolean-2d-array-indexing-behavior-in-numpy
"""

assert __name__ == "__main__"

import numpy as np

# TODO(eric): How to show this as copyable input/output REPL session?

def repl(expr):
    if expr == "":
        print()
    elif expr.startswith("#"):
        print(expr)
        return
    else:
        print(f">>> {expr}")
        if " = " in expr:
            exec(expr, globals(), globals())
        else:
            value = eval(expr, globals(), globals())
            if value is not None:
                print(repr(value))
                print()

lines = """
A = np.arange(9).reshape(3, 3)
A
# Slicing works as expected.
A[1:, :2]
# Indices that represent slice.
cols = [0, 1]
rows = [1, 2]
# Per OP, counterintuitively different.
A[rows, cols]
# Workaround: Select axes separately.
A[rows, :][:, cols]
""".strip().splitlines()
for line in lines:
    repl(line)
