"""
ROS geometry and visualization conversions for Drake.
"""

# ROS1 Messages.
from geometry_msgs.msg import Pose, Transform, TransformStamped
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import Marker, MarkerArray
# ROS1 API.
from rospy import Duration

from pydrake.common.eigen_geometry import Quaternion
from pydrake.geometry import (
    QueryObject,
    Role,
    SceneGraphInspector,
    FrameId,
    Shape,
    Box, Sphere, Cylinder, Mesh, Convex,
)
from pydrake.math import RigidTransform


def _write_pose_msg(X_AB, p, q):
    # p - position message
    # q - quaternion message
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
    # p - position message
    # q - quaternion message
    return RigidTransform(
        Quaternion(wxyz=[q.w, q.x, q.y, q.z]), [p.x, p.y, p.z])


def from_ros_pose(pose):
    """Converts ROS pose to Drake transform."""
    return _read_pose_msg(p=pose.position, q=pose.orientation)


def from_ros_transform(tr):
    """Converts ROS transform to Drake transform."""
    return _read_pose_msg(p=tr.translation, q=tr.rotation)


# N.B. Not yet used :(
DEFAULT_RGBA = [0.9, 0.9, 0.9, 1.0]


def to_ros_markers(shape, stamp, frame_name, X_FG, color):
    assert isinstance(shape, Shape), shape
    assert isinstance(frame_name, str), frame_name
    assert isinstance(X_FG, RigidTransform), X_FG
    marker = Marker()
    marker.header.stamp = stamp
    marker.header.frame_id = frame_name
    marker.pose = to_ros_pose(X_FG)
    marker.action = Marker.ADD
    marker.lifetime = Duration(0.)
    marker.frame_locked = True

    def use_color():
        # TODO: Colors are broken 'cause Geometry properties with Vector4 do not
        # play well with Python.
        assert color is None
        marker.color.a = 1.
        # marker.color.r, marker.color.g, marker.color.b, marker.color.a = color

    if type(shape) == Box:
        marker.type = Marker.CUBE
        marker.scale.x, marker.scale.y, marker.scale.z = shape.size()
        use_color()
        return [marker]
    if type(shape) == Sphere:
        marker.type = Marker.SPHERE
        marker.scale.x = shape.radius()
        marker.scale.y = shape.radius()
        marker.scale.z = shape.radius()
        use_color()
        return [marker]
    elif type(shape) == Cylinder:
        marker.type = Marker.CYLINDER
        marker.scale.x = shape.radius()
        marker.scale.y = shape.radius()
        marker.scale.z = shape.length()
        use_color()
        return [marker]
    elif type(shape) in [Mesh, Convex]:
        marker.type = Marker.MESH_RESOURCE
        marker.mesh_resource = f"file://{shape.filename()}"
        # TODO(eric.cousineau): Override this based on properties?
        marker.mesh_use_embedded_materials = True
        marker.scale.x, marker.scale.y, marker.scale.z = 3 * [shape.scale()]
        return [marker]
    else:
        assert False, f"Unsupported type: {shape}"


def get_role_properties(inspector, role, geometry_id):
    """Permits dynamic role retrival."""
    assert isinstance(inspector, SceneGraphInspector), inspector
    if role == Role.kProximity:
        return inspector.GetProximityProperties(geometry_id)
    elif role == Role.kIllustration:
        return inspector.GetIllustrationProperties(geometry_id)
    elif role == Role.kPerception:
        return inspector.GetPerceptionProperties(geometry_id)
    assert False, role


def to_ros_marker_array(query_object, role, stamp):
    assert isinstance(query_object, QueryObject), query_object
    assert isinstance(role, Role), role
    inspector = query_object.inspector()
    marker_array = MarkerArray()
    for geometry_id in inspector.GetAllGeometryIds():
        shape = inspector.GetShape(geometry_id)
        frame_id = inspector.GetFrameId(geometry_id)
        # TODO(eric.cousineau): Use X_FG.
        X_WG = query_object.X_WG(geometry_id)
        X_WF = query_object.X_WF(frame_id)
        X_FG = X_WF.inverse() @ X_WG
        frame_name = inspector.GetNameByFrameId(frame_id)
        properties = get_role_properties(inspector, role, geometry_id)
        if properties is None:
            # This role is not assigned for this geometry. Skip.
            continue
        # TODO(eric): Fix this :(
        # color = properties.GetPropertyOrDefault(
        #     "phong", "diffuse", DEFAULT_RGBA)
        color = None
        marker_array.markers += to_ros_markers(
            shape, stamp, frame_name, X_FG, color)
    # Ensure unique IDs.
    for i, marker in enumerate(marker_array.markers):
        # TODO(eric.cousineau): Reflect the namespace like Calder does for
        # collision, visual, etc.
        marker.id = i
    return marker_array


def to_ros_tf_message(query_object, stamp):
    assert isinstance(query_object, QueryObject), query_object
    inspector = query_object.inspector()
    frame_ids = set()
    for geometry_id in inspector.GetAllGeometryIds():
        frame_ids.add(inspector.GetFrameId(geometry_id))
    frame_ids = sorted(list(frame_ids))
    tf_message = TFMessage()
    for frame_id in frame_ids:
        frame_name = inspector.GetNameByFrameId(frame_id)
        X_WF = query_object.X_WF(frame_id)
        transform = TransformStamped()
        transform.header.frame_id = "world"
        transform.header.stamp = stamp
        transform.child_frame_id = frame_name
        transform.transform = to_ros_transform(X_WF)
        tf_message.transforms.append(transform)
    return tf_message
