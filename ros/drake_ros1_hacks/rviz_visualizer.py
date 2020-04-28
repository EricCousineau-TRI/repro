"""
Rviz Visualizer support in pydrake for ROS1.

Known issues:
* Colors cannot be retrieved in Python.
* (more to be noted)
"""

# ROS1 Messages.
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import MarkerArray
# ROS1 API.
import rospy

from pydrake.geometry import QueryObject, Role
from pydrake.systems.framework import (
    AbstractValue, LeafSystem, PublishEvent, TriggerType,
)

from drake_ros1_hacks.ros_geometry import (
    compare_marker_arrays,
    from_ros_pose,
    to_ros_marker_array,
    to_ros_pose,
    to_ros_tf_message,
    to_ros_transform,
)


class RvizVisualizer(LeafSystem):
    """
    Visualizes SceneGraph information in ROS1's Rviz.

    Input ports:
    - geometry_query: QueryObject
    Output ports: (none)

    ROS1 Subscribers: (none)
    ROS1 Publishers:
    - visualization_topic: MarkerArray
    - tf_topic: TFMessage

    Modeled after:
    - @calderpg-tri's SceneGraph code in TRI Anzu (private)
    - @gizatt's code in spartan:
        https://github.com/RobotLocomotion/spartan/blob/854b26e3a/src/catkin_projects/drake_iiwa_sim/src/ros_scene_graph_visualizer.cc
    - @russtedrake's MeshcatVisualizer in Drake:
        https://github.com/RobotLocomotion/drake/blob/6eabb61a/bindings/pydrake/systems/meshcat_visualizer.py

    However, this tries to do the "proper" approach by not using PoseBundle
    (nor hacks to try and 
    """

    def __init__(
            self,
            visualization_topic="/drake",
            period_sec=1./60,
            tf_topic="/tf",
            role=Role.kPerception):
        """
        Arguments:
            
            role: The Role to visualize geometry for.
        """
        LeafSystem.__init__(self)
        self._role = role

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

        # TODO(eric.cousineau): Rather than explicitly allocate publishers,
        # this should probably instead output the messages. Either together as a
        # tuple, or separately. Should delegate that to `ConnectRvizVisualizer`.
        self._marker_pub = rospy.Publisher(
            visualization_topic, MarkerArray, queue_size=1)
        self._tf_pub = rospy.Publisher(
            tf_topic, TFMessage, queue_size=1)

        self._marker_array_old = None
        self._marker_array_old_stamp = None

    def get_geometry_query_input_port(self):
        # We should standardize names...
        return self._geometry_query

    def _initialize(self, context, event):
        query_object = self._geometry_query.Eval(context)
        stamp = rospy.Time.now()
        marker_array = to_ros_marker_array(query_object, self._role, stamp)
        self._marker_array_old = marker_array
        self._marker_array_old_stamp = stamp
        self._marker_pub.publish(marker_array)
        # Initialize TF.
        tf_message = to_ros_tf_message(query_object, stamp)
        self._tf_pub.publish(tf_message)

    def _publish_tf(self, context, event):
        query_object = self._geometry_query.Eval(context)
        stamp = rospy.Time.now()
        tf_message = to_ros_tf_message(query_object, stamp)
        self._tf_pub.publish(tf_message)
        # We should have exactly the same message.
        # TODO(eric.cousineau): Support changing geometry, and remove this
        # method.
        # TODO(eric.cousineau): Is there a more SceneGraph-y way to do
        # geometry-change detection, without event hooks, and without explicit
        # serialization?
        old = self._marker_array_old
        new = to_ros_marker_array(
            query_object, self._role, stamp=self._marker_array_old_stamp)
        assert compare_marker_arrays(old, new), "Geometry changed!"


def ConnectRvizVisualizer(builder, scene_graph, **kwargs):
    """Connects an Rviz visualizer."""
    rviz = RvizVisualizer(**kwargs)
    builder.AddSystem(rviz)
    builder.Connect(
        scene_graph.get_query_output_port(),
        rviz.get_geometry_query_input_port())
    return rviz
