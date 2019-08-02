"""
Permits running Bazel's version of Jupyter directly.
"""

import os
import sys

from jupyter_bazel import jupyter_main

# Remove this so that the user can run notebooks with a normal PWD.
exit(jupyter_main(sys.argv[1:]))
