# -*- mode: python -*-
# vi: set ft=python :

load(
    "@bazel_ros2_rules//ros2:defs.bzl",
    "base_ros2_repository",
    "base_ros2_repository_attrs",
)
load(
    "//tools/external_data:external_data.bzl",
    "external_data_repository_attrs",
    "external_data_repository_download",
)
load(
    "@os//:os.bzl",
    "UBUNTU_RELEASE",
)

_OUTPUT = "ros2-linux"

# Archive relpath, Output
_ROS_ARCHIVES = {
    "20.04": [
        # Contained in ./ros2-linux
        ("ros2-humble-linux-focal-amd64-ci.tar.bz2", ""),
    ],
    "22.04": [
        # Contained in ./ros2-linux
        ("ros2-humble-linux-jammy-amd64-ci.tar.bz2", ""),
        # At root level.
        ("jammy-vision_opencv.tar", _OUTPUT),
    ],
}[UBUNTU_RELEASE]

def _workspace_label(relpath):
    return Label("//tools/workspace/ros2:" + relpath)

def _ros2_repository_impl(repo_ctx):
    repo_ctx.report_progress("Setting up ROS 2 workspace")

    file_relpaths = []
    for file_relpath, _ in _ROS_ARCHIVES:
        file_hash_relpath = file_relpath + ".sha512"
        repo_ctx.symlink(
            _workspace_label(file_hash_relpath),
            file_hash_relpath,
        )
        file_relpaths.append(file_relpath)

    # TODO(eric.cousineau): We should make this insenstive to
    # `.externa_data.yml` and `external_data`, ideally.
    external_data_repository_download(repo_ctx, file_relpaths)

    for file_relpath, output in _ROS_ARCHIVES:
        repo_ctx.extract(file_relpath, output = output)

    base_ros2_repository(
        repo_ctx,
        # This mapping makes sure files are referred to with relative paths
        # which is crucial for using native.glob()
        workspaces = {str(repo_ctx.path(_OUTPUT)): _OUTPUT},
    )

_ros2_repository_rule_attrs = {
    "_data": attr.label_list(
        default = [_workspace_label(x[0] + ".sha512") for x in _ROS_ARCHIVES],
    ),
}
_ros2_repository_rule_attrs.update(external_data_repository_attrs())
_ros2_repository_rule_attrs.update(base_ros2_repository_attrs())

_ros2_repository_rule = repository_rule(
    attrs = _ros2_repository_rule_attrs,
    implementation = _ros2_repository_impl,
    doc = """
Fetch a ROS 2 distribution hosted on Girder, extract it from the tarball,
scrape it, and bind it to a Bazel repository.
""",
    local = False,
)

# Please keep these sorted.
# Labels for all interface-only targets (including those that are OS specific)
# should be added to _ROS_INTERFACE_LABELS in
# //tools/skylark:anzu_ros_check.bzl
_ROS_PACKAGES = [
    "action_msgs",
    "builtin_interfaces",
    "cv_bridge",
    "geometry_msgs",
    "image_transport",
    "interactive_markers",
    "nav_msgs",
    "rclcpp",
    "rclcpp_action",
    "rclpy",
    "rmw_cyclonedds_cpp",
    "ros2cli_common_extensions",
    "rosbag2",
    "rosidl_default_generators",
    "rviz2",
    "sensor_msgs",
    "std_msgs",
    "std_srvs",
    "tf2_ros_py",
    "tf2_ros",
    "unique_identifier_msgs",
    "visualization_msgs",
]

def ros2_repository(name):
    _ros2_repository_rule(
        name = name,
        default_localhost_only = True,
        include_packages = _ROS_PACKAGES,
    )
