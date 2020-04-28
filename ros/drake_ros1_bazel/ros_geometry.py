"""
Simple ROS geometry utilities.
"""

import rospy
from geometry_msgs.msg import Pose, Transform

from pydrake.math import RigidTransform
from pydrake.common.eigen_geometry import Quaternion


def _write_pose_msg(X_AB, p, q):
    X_AB = RigidTransform(X_AB)
    p.x, p.y, p.z = X_AB.translation()
    q.w, q.x, q.y, q.z = X_AB.rotation().ToQuaternion().wxyz()


def to_ros_pose(X_AB):
    """Converts Drake transform to ROS pose."""
    msg = Pose()
    _write_pose_msg(X_AB, p=msg.position, q=msg.orientation)
    return msg


def to_ros_transform(X_AB):
    """Converts Drake transform to ROS transform."""
    msg = Transform()
    _write_pose_msg(X_AB, p=msg.translation, q=msg.rotation)
    return msg


def _read_pose_msg(p, q):
    return RigidTransform(
        Quaternion(wxyz=[q.w, q.x, q.y, q.z]), [p.x, p.y, p.z])


def from_ros_pose(pose):
    """Converts ROS pose to Drake transform."""
    return _read_pose_msg(p=pose.position, q=pose.orientation)


def from_ros_transform(tr):
    """Converts ROS transform to Drake transform."""
    return _read_pose_msg(p=tr.translation, q=tr.rotation)
