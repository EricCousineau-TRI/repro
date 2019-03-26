load("//tools/skylark/colcon:colcon.bzl", "colcon_repository")

def ros2_repository(name, hack_workspace_dir):
    colcon_repository(
        name = name,
        workspaces = [
            "./overlay",
            "/opt/ros/crystal",
        ],
        overlay_archives = {
            "./overlay": "file://{}/external/docker/overlay.tar.bz2".format(hack_workspace_dir),  # noqa
        },
        cc_packages = [
            "rclcpp",
            "std_msgs",
            "rmw_fastrtps_cpp",  # Specify RMW implementation.
        ],
        py_packages = [
            "rclpy",
            "std_msgs",
        ],
    )
