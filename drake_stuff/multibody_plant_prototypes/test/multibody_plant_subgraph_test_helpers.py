"""
Helpers for test code.
"""

from collections import defaultdict
import random

import numpy as np
from numpy.testing import assert_array_equal

from pydrake.geometry import Box, GeometryInstance, HalfSpace
from pydrake.math import RigidTransform
from pydrake.multibody.plant import CoulombFriction
from pydrake.multibody.tree import (
    BallRpyJoint,
    FixedOffsetFrame,
    PrismaticJoint,
    RevoluteJoint,
    SpatialInertia,
    UnitInertia,
    UniversalJoint,
    WeldJoint,
)
from pydrake.systems.primitives import ConstantVectorSource

from multibody_plant_prototypes.containers import strict_zip

from .. import multibody_extras as me
from .. import multibody_plant_subgraph as mut

JOINT_CLS_LIST = [
    PrismaticJoint,
    RevoluteJoint,
    BallRpyJoint,
    UniversalJoint,
    WeldJoint,
]


def build_with_no_control(builder, plant, model):
    """Connects a zero-torque input to a given model instance in a plant."""
    # TODO(eric.cousineau): Use `multibody_plant_prototypes.control` if the dependency can
    # be simplified.
    nu = plant.num_actuated_dofs(model)
    constant = builder.AddSystem(ConstantVectorSource(np.zeros(nu)))
    builder.Connect(
        constant.get_output_port(0),
        plant.get_actuation_input_port(model))


def compare_frame_poses(
        plant, context, sub_plant, sub_context,
        base_frame_name, test_frame_name, **kwargs):
    """Compares the poses of two frames."""
    X_BT_sub = sub_plant.CalcRelativeTransform(
        sub_context,
        sub_plant.GetFrameByName(base_frame_name),
        sub_plant.GetFrameByName(test_frame_name))
    X_BT = plant.CalcRelativeTransform(
        context,
        plant.GetFrameByName(base_frame_name, **kwargs),
        plant.GetFrameByName(test_frame_name, **kwargs))
    np.testing.assert_allclose(
        X_BT_sub.GetAsMatrix4(),
        X_BT.GetAsMatrix4(), rtol=0., atol=1e-10)


def add_arbitrary_multibody_stuff(
        plant, num_bodies=30, slight_difference=False):
    """
    Deterministic, physically valid, jumble of arbitrary stuff.

    The goal of this factory is to:

    - Produce a set of elements and cases that exercise each code path in
      `MultibodyPlantSubgraph.add_to` and `MultibodyPlantElementsMap`.
    - Ensure each element has somewhat "random" but unique properties for each
      element produced.
    - Allow for a slight difference to show that strict plant comparison can be
      falsified.
    """
    count = defaultdict(lambda: 0)

    def i_next(key=None):
        # Increments for a given key.
        count[key] += 1
        return count[key]

    def maybe():
        # Returns True or False randomly.
        return random.choice([True, False])

    def random_model_instance():
        # Returns a "random" model instance (by incrementing).
        i = i_next()
        return plant.AddModelInstance(f"model_{i}")

    def random_position():
        # Returns a random position.
        return [0.2 * random.random(), 0, 0]

    def random_X():
        # Returns a random pose.
        return RigidTransform(random_position())

    def random_body():
        # Returns a random body, with an incrementing name.
        inertia = SpatialInertia(
            mass=random.uniform(0.2, 1.),
            p_PScm_E=random_position(),
            G_SP_E=UnitInertia(
                Ixx=random.uniform(0.2, 0.3),
                Iyy=random.uniform(0.2, 0.3),
                Izz=random.uniform(0.2, 0.3),
            ),
        )
        return plant.AddRigidBody(
            name=f"body_{i_next()}",
            M_BBo_B=inertia,
            model_instance=random_model_instance(),
        )

    def random_frame(parent_frame):
        # Returns a random frame, with an incrementing name.
        i = i_next()
        return plant.AddFrame(FixedOffsetFrame(
            name=f"frame_{i}", P=parent_frame,
            X_PF=random_X(),
            model_instance=parent_frame.model_instance(),
        ))

    def random_joint(parent, child):
        # Returns a random joint, but with an incrementing name. Note that we
        # use a separate index so that we ensure we can loop through all
        # joints.
        i = i_next("joint")
        name = f"joint_{i}"
        joint_cls = JOINT_CLS_LIST[i % len(JOINT_CLS_LIST)]
        frame_on_parent = random_frame(parent.body_frame())
        frame_on_child = random_frame(child.body_frame())
        axis = np.zeros(3)
        axis[i_next() % 3] = 1
        damping = random.random()
        if joint_cls == BallRpyJoint:
            joint = BallRpyJoint(
                name,
                frame_on_parent=frame_on_parent,
                frame_on_child=frame_on_child,
                damping=damping,
            )
        elif joint_cls == PrismaticJoint:
            joint = PrismaticJoint(
                name,
                frame_on_parent=frame_on_parent,
                frame_on_child=frame_on_child,
                axis=axis,
                damping=damping,
            )
        elif joint_cls == RevoluteJoint:
            joint = RevoluteJoint(
                name,
                frame_on_parent=frame_on_parent,
                frame_on_child=frame_on_child,
                axis=axis,
                damping=damping,
            )
        elif joint_cls == UniversalJoint:
            joint = UniversalJoint(
                name,
                frame_on_parent=frame_on_parent,
                frame_on_child=frame_on_child,
                damping=damping,
            )
        elif joint_cls == WeldJoint:
            joint = WeldJoint(
                name,
                frame_on_parent_P=frame_on_parent,
                frame_on_child_C=frame_on_child,
                X_PC=random_X(),
            )
        else:
            assert False
        return plant.AddJoint(joint)

    def random_joint_actuator(joint):
        # Creates a random joint actuator.
        assert joint is not None
        i = i_next()
        return plant.AddJointActuator(
            f"actuator_{i}", joint, effort_limit=random.uniform(1, 2))

    def random_geometry(body):
        # Creates a random geometry.
        i = i_next()
        box = Box(
            width=random.uniform(0.1, 0.3),
            depth=random.uniform(0.1, 0.3),
            height=random.uniform(0.1, 0.3),
        )
        plant.RegisterVisualGeometry(
            body=body,
            X_BG=random_X(),
            shape=box,
            name=f"visual_{i}",
            diffuse_color=[random.random(), 0, 0, 0.75],
        )
        static_friction = random.uniform(0.1, 1.)
        plant.RegisterCollisionGeometry(
            body=body,
            X_BG=random_X(),
            shape=box,
            name=f"collision_{i}",
            coulomb_friction=CoulombFriction(
                static_friction=static_friction,
                dynamic_friction=static_friction / 2,
            )
        )

    # Add ground plane.
    X_FH = HalfSpace.MakePose([0, 0, 1], [0, 0, 0])
    plant.RegisterCollisionGeometry(
        plant.world_body(), X_FH, HalfSpace(), "ground_plane_collision",
        CoulombFriction(0.8, 0.3))

    grid_rows = 5
    prev_body = None
    for i in range(num_bodies):
        random.seed(i)
        body = random_body()
        grid_col = i % grid_rows
        grid_row = int(i / grid_rows)
        if slight_difference:
            grid_row += 1
        plant.SetDefaultFreeBodyPose(
            body, RigidTransform([grid_col, grid_row, 2]))
        random_frame(body.body_frame())
        # Consider attaching a joint and/or frame to the world.
        if maybe() or num_bodies < 3:
            prev_body = plant.world_body()
            random_frame(plant.world_frame())
        if prev_body is not None and (maybe() or num_bodies < 3):
            joint = random_joint(prev_body, body)
            if joint.num_velocities() == 1 and (maybe() or num_bodies < 3):
                random_joint_actuator(joint)
        if plant.geometry_source_is_registered():
            random_geometry(body)
        prev_body = body


def assert_inertia_equals(a, b):
    assert_array_equal(a.CopyToFullMatrix6(), b.CopyToFullMatrix6())


def assert_pose_equals(a, b):
    assert_array_equal(a.GetAsMatrix4(), b.GetAsMatrix4())


def check_element(a, b, check_index=True):
    """Checks that two multibody elements have similar base properties."""
    assert a is not b
    if check_index:
        assert a.index() == b.index()
    assert a.name() == b.name(), (a.name(), b.name())
    assert type(a) == type(b)
    assert a.model_instance() == b.model_instance()


def assert_shape_equals(a, b):
    assert type(a) == type(b)
    if type(a) == Box:
        assert a.width() == b.width()
        assert a.height() == b.height()
        assert a.depth() == b.depth()
    elif type(a) == HalfSpace:
        pass
    else:
        assert False


def assert_value_equals(value_a, value_b):
    a = value_a.get_value()
    b = value_b.get_value()
    assert type(a) == type(b)
    if type(a) == CoulombFriction:
        assert a.static_friction() == b.static_friction()
        assert a.dynamic_friction() == b.dynamic_friction()
    else:
        assert a == b


def assert_properties_equals(prop_a, prop_b):
    if prop_a is None:
        assert prop_b is None
        return
    groups = prop_a.GetGroupNames()
    assert groups == prop_b.GetGroupNames()
    for group_name in groups:
        group_a = prop_a.GetPropertiesInGroup(group_name)
        group_b = prop_b.GetPropertiesInGroup(group_name)
        assert len(group_a) == len(group_b)
        for name, value_a in group_a.items():
            value_b = group_b[name]
            try:
                assert_value_equals(value_a, value_b)
            except RuntimeError as e:
                if "AddValueInstantiation" in str(e):
                    # TODO(eric.cosuineau): Fix this stuff for Vector4d.
                    assert (group_name, name) == ("phong", "diffuse")
                else:
                    raise


def assert_plant_equals(plant_a, scene_graph_a, plant_b, scene_graph_b):
    """
    Asserts that two plants are (almost) completely equal; more specifically:
    - All model instances, bodies, joints, and joint actuators have the same
      indices.
        - Frames may have different indices, due to ordering.
    - All properties of each element are "exactly" the same.
    """
    assert plant_a is not plant_b
    if scene_graph_b is not None:
        assert scene_graph_a is not None
    elem_a = mut.get_elements_from_plant(plant_a, scene_graph_a)
    checked_a = mut.MultibodyPlantElements(plant_a, scene_graph_a)
    elem_b = mut.get_elements_from_plant(plant_b, scene_graph_b)
    checked_b = mut.MultibodyPlantElements(plant_b, scene_graph_b)

    def assert_body_equals(body_a, body_b):
        check_element(body_a, body_b)
        assert_inertia_equals(
            body_a.default_spatial_inertia(), body_b.default_spatial_inertia())
        assert body_a.model_instance() in checked_a.model_instances
        assert body_b.model_instance() in checked_b.model_instances
        assert_pose_equals(
            plant_a.GetDefaultFreeBodyPose(body_a),
            plant_b.GetDefaultFreeBodyPose(body_b))
        checked_a.bodies.add(body_a)
        checked_b.bodies.add(body_b)

    def assert_frame_equals(frame_a, frame_b):
        check_element(frame_a, frame_b, check_index=False)
        assert frame_a.body() in checked_a.bodies
        assert frame_b.body() in checked_b.bodies
        assert_pose_equals(
            frame_a.GetFixedPoseInBodyFrame(),
            frame_b.GetFixedPoseInBodyFrame())
        checked_a.frames.add(frame_a)
        checked_b.frames.add(frame_b)

    def assert_joint_equals(joint_a, joint_b):
        check_element(joint_a, joint_b)
        assert joint_a.frame_on_parent() in checked_a.frames
        assert joint_b.frame_on_parent() in checked_b.frames
        assert joint_a.frame_on_child() in checked_a.frames
        assert joint_b.frame_on_child() in checked_b.frames
        assert_array_equal(
            joint_a.position_lower_limits(),
            joint_b.position_lower_limits())
        assert_array_equal(
            joint_a.position_upper_limits(),
            joint_b.position_upper_limits())
        assert_array_equal(
            joint_a.velocity_upper_limits(),
            joint_b.velocity_upper_limits())
        assert_array_equal(
            joint_a.velocity_lower_limits(),
            joint_b.velocity_lower_limits())
        assert_array_equal(
            joint_a.acceleration_lower_limits(),
            joint_b.acceleration_lower_limits())
        assert_array_equal(
            joint_a.acceleration_upper_limits(),
            joint_b.acceleration_upper_limits())
        assert_array_equal(
            joint_a.default_positions(),
            joint_b.default_positions())
        joints_with_damping = (
            PrismaticJoint,
            RevoluteJoint,
            BallRpyJoint,
            UniversalJoint,
        )
        if type(joint_a) in joints_with_damping:
            assert joint_a.damping() == joint_b.damping()
        if type(joint_a) == PrismaticJoint:
            assert_array_equal(
                joint_a.translation_axis(),
                joint_b.translation_axis())
        if type(joint_a) == RevoluteJoint:
            assert_array_equal(
                joint_a.revolute_axis(), joint_b.revolute_axis())
        if type(joint_a) == WeldJoint:
            assert_pose_equals(joint_a.X_PC(), joint_b.X_PC())
        checked_a.joints.add(joint_a)
        checked_b.joints.add(joint_b)

    def assert_geometry_equals(a, b):
        inspector_a = scene_graph_a.model_inspector()
        body_a = plant_a.GetBodyFromFrameId(inspector_a.GetFrameId(a))
        assert body_a in checked_a.bodies
        geometry_a = inspector_a.CloneGeometryInstance(a)
        inspector_b = scene_graph_b.model_inspector()
        body_b = plant_b.GetBodyFromFrameId(inspector_b.GetFrameId(b))
        assert body_b in checked_b.bodies
        geometry_b = inspector_b.CloneGeometryInstance(b)
        assert geometry_a.name() == geometry_b.name(), (
            geometry_a.name(), geometry_b.name())
        assert_pose_equals(geometry_a.pose(), geometry_b.pose())
        assert_shape_equals(
            geometry_a.release_shape(), geometry_b.release_shape())
        prop_funcs = [
            GeometryInstance.perception_properties,
            GeometryInstance.proximity_properties,
            GeometryInstance.illustration_properties,
        ]
        for prop_func in prop_funcs:
            assert_properties_equals(
                prop_func(geometry_a), prop_func(geometry_b))

    def frame_map(frames):
        out = defaultdict(set)
        for frame in frames:
            # Some frames may not have a name :(
            key = (frame.body().name(), frame.name())
            out[key].add(frame)
        return out

    for a, b in strict_zip(elem_a.model_instances, elem_b.model_instances):
        assert a is not b
        assert a == b
        checked_a.model_instances.add(a)
        checked_b.model_instances.add(b)

    for body_a, body_b in strict_zip(elem_a.bodies, elem_b.bodies):
        assert_body_equals(body_a, body_b)

    # N.B. Because frame indices can be shifted when adding bodies, we cannot
    # trust this ordering. Instead, we need to find an identifier.
    frame_map_a = frame_map(elem_a.frames)
    frame_map_b = frame_map(elem_b.frames)
    assert len(frame_map_a) == len(frame_map_b)
    for key, frames_a in frame_map_a.items():
        frames_b = frame_map_b[key]
        for frame_a, frame_b in strict_zip(frames_a, frames_b):
            assert_frame_equals(frame_a, frame_b)

    for joint_a, joint_b in strict_zip(elem_a.joints, elem_b.joints):
        assert_joint_equals(joint_a, joint_b)

    cur_iter = strict_zip(elem_a.joint_actuators, elem_b.joint_actuators)
    for joint_actuator_a, joint_actuator_b in cur_iter:
        check_element(joint_actuator_a, joint_actuator_b)
        assert (
            joint_actuator_a.effort_limit() == joint_actuator_b.effort_limit())

    if scene_graph_b is not None:
        cur_iter = strict_zip(elem_a.geometry_ids, elem_b.geometry_ids)
        for geometry_id_a, geometry_id_b in cur_iter:
            assert_geometry_equals(geometry_id_a, geometry_id_b)
