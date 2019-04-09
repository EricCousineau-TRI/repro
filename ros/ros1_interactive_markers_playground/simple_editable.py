#!/usr/bin/env python3

# Following: https://github.com/ros-visualization/visualization_tutorials/blob/indigo-devel/interactive_marker_tutorials/scripts/basic_controls.py
# Borrowing from: https://github.com/WPI-ARC/teleop_examples/blob/f4fad4c/hubo_marker_teleop/src/hubo_marker_teleop/gt_hubo_marker_teleop.py

import copy
import math
import time

import rospy
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from interactive_markers.menu_handler import MenuHandler
from visualization_msgs.msg import Marker, InteractiveMarker, InteractiveMarkerControl, InteractiveMarkerFeedback
from geometry_msgs.msg import Point, Quaternion, Pose

DT_DOUBLE_CLICK = 0.25


def assign(obj=None, **kwargs):
    if obj is None:
        # Consider returning functor.
        return dict(kwargs)
    for prop, value in kwargs.items():
        if isinstance(value, dict):
            assign(getattr(obj, prop), **value)
        else:
            setattr(obj, prop, value)
    return obj


def make_6dof_controls():
    # Edit axes.
    # N.B. Follow order from tutorial?
    n = 1 / math.sqrt(2)
    axes = [
        ('x', assign(Quaternion(), w=n, x=n, y=0, z=0)),
        ('z', assign(Quaternion(), w=n, x=0, y=0, z=n)),
        ('y', assign(Quaternion(), w=n, x=0, y=n, z=0)),
    ]
    controls = []
    for name, quat in axes:
        control = InteractiveMarkerControl()
        control.orientation = quat
        control.name = "rotate_" + name
        control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        controls.append(control)
        control = InteractiveMarkerControl()
        control.orientation = quat
        control.name = "move_" + name
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        controls.append(control)
    return controls


def noop(*args, **kwargs):
    pass


class SimpleInteractiveMarker(object):
    """Creates a group of markers whose poses can be edited."""
    def __init__(self, server, name, X_WM, markers):
        self._server = server
        self.name = name
        self._X_WM = X_WM
        self._menu = MenuHandler()
        self._edit_id = self._menu.insert("Edit", callback=self._edit_callback)
        self._delete_id = self._menu.insert("Delete", callback=self._delete_callback)
        self._menu.insert(self.name, callback=noop)  # For annotation.
        self._t_click_prev = -float('inf')
        # Make template marker.
        template = InteractiveMarker()
        template.name = self.name
        template.description = name
        template.header.frame_id = "world"
        viz = InteractiveMarkerControl()
        viz.interaction_mode = InteractiveMarkerControl.BUTTON
        viz.always_visible = True
        viz.markers.extend(markers)
        template.controls.append(viz)
        self._template = template
        # Initialize.
        self.reset()

    def get_pose(self):
        return self._X_WM

    def set_pose(self, X_WM):
        self._X_WM = X_WM
        self._server.setPose(self.name, self._X_WM)
        self._server.applyChanges()

    def reset(self):
        self._edit = False
        self._update()

    def clear(self):
        self._server.erase(self.name)
        self._server.applyChanges()

    def _update(self):
        # Construct interactive marker.
        marker = copy.deepcopy(self._template)
        marker.pose = self._X_WM
        if self._edit:
            marker.controls += make_6dof_controls()
            state = MenuHandler.CHECKED
        else:
            state = MenuHandler.UNCHECKED
        self._menu.setCheckState(self._edit_id, state)
        self._server.insert(marker, self._marker_callback)
        self._menu.apply(self._server, marker.name)
        self._server.applyChanges()

    def _marker_callback(self, msg):
        if msg.event_type == InteractiveMarkerFeedback.POSE_UPDATE:
            self._X_WM = msg.pose
        if msg.event_type == InteractiveMarkerFeedback.BUTTON_CLICK:
            # Require double-click.
            t_click = time.time()
            dt_click = t_click - self._t_click_prev
            self._t_click_prev = t_click
            if dt_click < DT_DOUBLE_CLICK:
                self._edit = not self._edit
                self._update()

    def _edit_callback(self, _):
        # Toggle.
        self._edit = not self._edit
        self._update()

    def _delete_callback(self, _):
        self.clear()


def main():
    rospy.init_node("rviz_stuff")
    server = InteractiveMarkerServer("rviz_stuff")
    cube = assign(
        Marker(), type=Marker.CUBE,
        scale=assign(x=0.1, y=0.1, z=0.1),
        color=assign(r=1, g=0, b=0, a=1),
        pose=assign(Pose(), position=assign(x=0.1, y=0, z=0)),
    )
    sphere = assign(
        Marker(), type=Marker.SPHERE,
        scale=assign(x=0.1, y=0.1, z=0.1),
        color=assign(r=1, g=0, b=0, a=1),
        pose=assign(Pose(), position=assign(x=-0.1, y=0.1, z=0)),
    )
    X_WA = assign(Pose(), position=assign(x=1, y=0, z=1), orientation=assign(w=1, x=0, y=0, z=0))
    a = SimpleInteractiveMarker(server, "a", X_WA, [cube])
    X_WB = assign(Pose(), position=assign(x=1, y=1, z=2), orientation=assign(w=1, x=0, y=0, z=0))
    b = SimpleInteractiveMarker(server, "b", X_WB, [cube, sphere])
    b.set_pose(assign(Pose(), position=assign(x=-1)))
    X_WC = Pose()
    c = SimpleInteractiveMarker(server, "c", X_WC, [sphere])
    a.clear()
    a.reset()
    rospy.spin()


if __name__ == "__main__":
    main()
