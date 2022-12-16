"""
Partially exported example of defining a wrapper mechanism (non-Systems), but 
then adding that to a System.
"""

from types import SimpleNamespace

import rclpy
from rclpy.executors import SingleThreadedExecutor

from pydrake.common.value import Value
from pydrake.math import RigidTransform
from pydrake.multibody.math import SpatialForce, SpatialVelocity
from pydrake.systems.framework import LeafSystem

from CODE_NOT_EXPORTED_YET_SORRY import (
    ExampleRosDeviceCommand,
    ExampleRosDeviceStatus,
    PoseOutputPorts,
    declare_pose_inputs,
    from_ros_pose,
    to_ros_pose,
    to_ros_twist,
    to_ros_wrench,
)
from systems_controllers_rough_example.helpers import (
    simple_cache_declare,
    simple_cache_ensure_init,
)


class BaseHapticWrapper:
    """Base wrapper class to be used by HapticDeviceSystem."""

    def reset(self):
        """Returns pose."""
        # TODO(eric.cousineau): Return message too?
        raise NotImplementedError()

    def read(self):
        """Returns pose, message."""
        raise NotImplementedError()

    def send_force(self, F):
        """Sends force."""
        raise NotImplementedError()

    def send_pose(self, X, V, gains):
        """Sends pose (and velocity)."""
        raise NotImplementedError()


class DummyHapticWrapper(BaseHapticWrapper):
    def __init__(self):
        self._X = RigidTransform()
        self._message = object()

    def reset(self):
        return self._X

    def read(self):
        return self._X, self._message

    def send_force(self, F):
        pass

    def send_pose(self, X, V, gains):
        pass


class ExampleRosDeviceWrapper(BaseHapticWrapper):
    def __init__(
        self,
        *,
        topic_prefix="/example_ros_device",
        node_name="example_ros_device_wrapper",
    ):
        self._node = rclpy.create_node(node_name)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._sub = self._node.create_subscription(
            ExampleRosDeviceStatus,
            f"{topic_prefix}/status",
            self._on_status,
            qos_profile=1,
        )
        self._pub = self._node.create_publisher(
            ExampleRosDeviceCommand,
            f"{topic_prefix}/command",
            qos_profile=1,
        )
        self._pose = None

    def _on_status(self, message):
        self._pose = from_ros_pose(message.pose)
        self._message = message

    def reset(self):
        self._pose = None
        while self._pose is None:
            self._executor.spin_once(timeout_sec=0.001)
        return self._pose

    def read(self):
        self._executor.spin_once(timeout_sec=0.0001)
        return self._pose, self._message

    def send_force(self, F):
        command = ExampleRosDeviceCommand()
        command.header.stamp = self._node.get_clock().now().to_msg()
        command.provides_wrench = True
        command.desired_wrench = to_ros_wrench(F)
        self._pub.publish(command)

    def send_pose(self, X, V, gains):
        command = ExampleRosDeviceCommand()
        command.header.stamp = self._node.get_clock().now().to_msg()
        command.provides_pose = True
        command.kp_translation = gains.kp_translation
        command.kd_translation = gains.kd_translation
        command.kp_rotation = gains.kp_rotation
        command.kd_rotation = gains.kd_rotation
        command.desired_pose = to_ros_pose(X)
        command.desired_twist = to_ros_twist(V)
        self._pub.publish(command)


def declare_pose_outputs(
    system,
    frames,
    *,
    name_X,
    calc_X,
    name_V,
    calc_V,
    prerequisites_of_calc=None,
):
    alloc_X = Value[RigidTransform]
    alloc_V = Value[SpatialVelocity]
    if prerequisites_of_calc is None:
        prerequisites_of_calc = {system.all_sources_ticket()}
    return PoseOutputPorts(
        frames=frames,
        X=system.DeclareAbstractOutputPort(
            name_X,
            alloc=alloc_X,
            calc=calc_X,
            prerequisites_of_calc=prerequisites_of_calc,
        ),
        V=system.DeclareAbstractOutputPort(
            name_V,
            alloc=alloc_V,
            calc=calc_V,
            prerequisites_of_calc=prerequisites_of_calc,
        ),
    )


class HapticDeviceSystem(LeafSystem):
    """
    Haptic device LeafSystem interface operating in pose- (X, V) and/or wrench-
    (F) space.

     WARNING:
        `wrapper` can be stateful, thus can only operate on a single context at
        a given time! Concretely, don't allocate two separate contexts and use
        this system (or `wrapper` reference) and interleave evaluating stuff
        against those contexts with this system.
    """

    def __init__(
        self,
        *,
        frames,
        wrapper,
        on_message,
        period_sec,
        spatial_gains,
    ):
        super().__init__()
        assert isinstance(wrapper, BaseHapticWrapper)

        def calc_cache(context, cache):
            simple_cache_ensure_init(context, cache, on_init)

        cache_entry = simple_cache_declare(self, calc_cache)

        def on_init(context, cache):
            # TODO(eric.cousinaeu): Use haptic device name.
            print("HapticDevice: Reset")
            cache.init.X_WH = wrapper.reset()
            print("  Running")

        state_init = SimpleNamespace(X_WH=None)
        state_index = self.DeclareAbstractState(Value[object](state_init))

        # Inputs.
        self.inputs_pose_desired = declare_pose_inputs(
            self,
            frames,
            name_X="X_WHdes",
            name_V="V_WHdes",
        )
        self.input_force_desired = self.DeclareAbstractInputPort(
            "F_WH", Value[SpatialForce]()
        )

        def on_discrete_update(context, raw_state):
            abstract_state = raw_state.get_mutable_abstract_state(state_index)
            state = abstract_state.get_mutable_value()
            # Retrieve state.
            X_WH, message = wrapper.read()
            state.X_WH = X_WH
            on_message(wrapper, message)
            # Maybe send commands.
            has_pose = self.inputs_pose_desired.has_value(context)
            has_force = self.input_force_desired.HasValue(context)
            # TODO(eric.cousineau): Support hybrid?
            assert not (has_pose and has_force)
            if has_pose:
                X_WHdes, V_WHdes = self.inputs_pose_desired.eval(context)
                wrapper.send_pose(X_WHdes, V_WHdes, spatial_gains)
            elif has_force:
                F_WH = self.input_force_desired.Eval(context)
                wrapper.send_force(F_WH)

        # Discrete update.
        self.DeclarePeriodicUnrestrictedUpdateEvent(
            period_sec=period_sec,
            offset_sec=0.0,
            update=on_discrete_update,
        )

        # Outputs.
        # TODO(eric.cousineau): Output force?
        def get_pose(context):
            cache = cache_entry.Eval(context)
            state = context.get_abstract_state(state_index).get_value()
            if state.X_WH is None:
                return cache.init.X_WH
            else:
                return state.X_WH

        def calc_actual_X(context, output):
            X_WH = get_pose(context)
            output.set_value(X_WH)

        def calc_actual_V(context, output):
            # TODO(eric.cousineau): Be more intelligent.
            V_WH = SpatialVelocity.Zero()
            output.set_value(V_WH)

        self.outputs_pose_actual = declare_pose_outputs(
            self,
            frames,
            name_X="X_WH",
            calc_X=calc_actual_X,
            name_V="V_WH",
            calc_V=calc_actual_V,
            prerequisites_of_calc={self.abstract_state_ticket(state_index)},
        )
