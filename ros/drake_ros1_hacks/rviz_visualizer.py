"""
Rviz Visualizer support in pydrake for ROS1.

Known issues:
* Colors cannot be retrieved in Python.
* (more to be noted)
"""

import rospy
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from pydrake.geometry import (
    QueryObject,
    Role,
    Box, Sphere, Cylinder, Mesh, Convex,
)
from pydrake.systems.framework import (
    AbstractValue, LeafSystem, PublishEvent, TriggerType,
)

from drake_ros1_hacks.ros_geometry import to_ros_pose, to_ros_transform

DEFAULT_RGBA = [0.9, 0.9, 0.9, 1.0]


def to_markers(shape, stamp, frame_id, X_FG, color):
    marker = Marker()
    marker.header.stamp = stamp
    marker.header.frame_id = frame_id
    marker.pose = to_ros_pose(X_FG)
    marker.action = Marker.ADD
    marker.lifetime = rospy.Duration(0.)
    marker.frame_locked = True

    # TODO: Colors are broken 'cause Geometry properties with Vector4 do not
    # play well with Python.
    assert color is None
    marker.color.a = 1.
    # marker.color.r, marker.color.g, marker.color.b, marker.color.a = color

    if type(shape) == Box:
        marker.type = Marker.CUBE
        marker.scale.x, marker.scale.y, marker.scale.z = shape.size()
        return [marker]
    if type(shape) == Sphere:
        marker.type = Marker.SPHERE
        marker.scale.x = shape.radius()
        marker.scale.y = shape.radius()
        marker.scale.z = shape.radius()
        return [marker]
    elif type(shape) == Cylinder:
        marker.type = Marker.CYLINDER
        marker.scale.x = shape.radius()
        marker.scale.y = shape.radius()
        marker.scale.z = shape.length()
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
    if role == Role.kProximity:
        return inspector.GetProximityProperties(geometry_id)
    elif role == Role.kIllustration:
        return inspector.GetIllustrationProperties(geometry_id)
    elif role == Role.kPerception:
        return inspector.GetPerceptionProperties(geometry_id)
    assert False, role


def to_marker_array(query_object, role):
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
        # TODO(eric): Fix this :(
        # color = properties.GetPropertyOrDefault(
        #     "phong", "diffuse", DEFAULT_RGBA)
        marker_array.markers += to_markers(
            shape, rospy.Time.now(), frame_name, X_FG, color=None)
    # Ensure unique IDs.
    for i, marker in enumerate(marker_array.markers):
        marker.id = i
    return marker_array


def to_tf_message(query_object):
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
        transform.header.stamp = rospy.Time.now()
        transform.child_frame_id = frame_name
        transform.transform = to_ros_transform(X_WF)
        tf_message.transforms.append(transform)
    return tf_message


class RvizVisualizer(LeafSystem):
    # Better architecture: Output visualization + TF messages on port, let it
    # be published separately.
    # Modeled after Calder's code in Anzu, and Greg's code in spartan.
    def __init__(
            self,
            scene_graph,
            topic="/drake",
            period_sec=1./60,
            tf_topic="/tf",
            role=Role.kPerception):
        LeafSystem.__init__(self)
        self._scene_graph = scene_graph
        self._role = role

        self._marker_pub = rospy.Publisher(topic, MarkerArray, queue_size=1)
        self._tf_pub = rospy.Publisher(tf_topic, TFMessage, queue_size=1)

        self._geometry_query = self.DeclareAbstractInputPort(
            "geometry_query", AbstractValue.Make(QueryObject()))
        self.DeclareInitializationEvent(
            event=PublishEvent(
                trigger_type=TriggerType.kInitialization,
                callback=self._initialize))
        self.DeclarePeriodicEvent(
            period_sec=period_sec,
            offset_sec=0.,
            event=PublishEvent(
                trigger_type=TriggerType.kPeriodic,
                callback=self._publish_tf))

    def get_geometry_query_input_port(self):
        return self._geometry_query

    def _initialize(self, context, event):
        query_object = self._geometry_query.Eval(context)
        marker_array = to_marker_array(query_object, self._role)
        self._marker_pub.publish(marker_array)
        tf_message = to_tf_message(query_object)
        self._tf_pub.publish(tf_message)

    def _publish_tf(self, context, event):
        # N.B. This does not check for changes in geometry.
        query_object = self._geometry_query.Eval(context)
        tf_message = to_tf_message(query_object)
        self._tf_pub.publish(tf_message)


def ConnectRvizVisualizer(builder, scene_graph, **kwargs):
    rviz = RvizVisualizer(scene_graph, **kwargs)
    builder.AddSystem(rviz)
    builder.Connect(
        scene_graph.get_query_output_port(),
        rviz.get_geometry_query_input_port())
    return rviz
