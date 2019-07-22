# ROS1 Python Dependencies

**NOTE**: This is used to generate an overlay of ROS1 C extension modules for
Python 3 for consumption by Bazel. This may require modification for other use
cases.

To incorporate a C extension module / package:

* Add it to `PY3_REBUILD` in the module list
* Add the relevant repository to `./do_py3_rebuild.repo`
* Ensure you have Docker installed. See Docker website instructions.
* Run `./do_py3_rebuild.py`. This will output an archive with some of the
overlay files.
