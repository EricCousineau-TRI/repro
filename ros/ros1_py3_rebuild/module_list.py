# This file is intended to be used by both Bazel and Python; this excerpt is
# only for an example, and thus comments out stuff that isn't immediately used.

# # Modules that are natively Python 3 compatible.
# PY3_COMPATIBLE = [
#     "gencpp",
#     "genmsg",
#     "geometry_msgs",
#     "interactive_markers",
#     "rospy",
#     "std_msgs",
#     "tf2_ros",
#     "visualization_msgs",
# ]

# Modules that must be rebuilt explicitly for Python 3. Review the README
# before updating this.
PY3_REBUILD = [
    "cv_bridge",
    "tf2_py",
]
