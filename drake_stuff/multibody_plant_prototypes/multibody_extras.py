import numpy as np

from pydrake.geometry import Role
from pydrake.multibody.math import SpatialVelocity
from pydrake.multibody.tree import (
    BodyIndex,
    FrameIndex,
    JacobianWrtVariable,
    JointActuatorIndex,
    JointIndex,
    ModelInstanceIndex,
)

from multibody_plant_prototypes.cc import GetGeometries, RemoveRoleFromGeometries
from multibody_plant_prototypes.containers import take_first


def _get_plant_aggregate(num_func, get_func, index_cls, model_instances=None):
    items = []
    for i in range(num_func()):
        item = get_func(index_cls(i))
        if model_instances is None or item.model_instance() in model_instances:
            items.append(item)
    return items


def get_model_instances(plant):
    # TODO(eric.cousineau): Hoist this somewhere?
    return _get_plant_aggregate(
        plant.num_model_instances, lambda x: x,
        ModelInstanceIndex)


def get_bodies(plant, model_instances=None):
    # TODO(eric.cousineau): Hoist this somewhere?
    return _get_plant_aggregate(
        plant.num_bodies, plant.get_body, BodyIndex, model_instances)


def get_frames(plant, model_instances=None):
    # TODO(eric.cousineau): Hoist this somewhere?
    return _get_plant_aggregate(
        plant.num_frames, plant.get_frame, FrameIndex, model_instances)


def get_frames_attached_to(plant, bodies):
    # TODO(eric.cousineau): Hoist this somewhere?
    frames = []
    for frame in get_frames(plant):
        if frame.body() in bodies:
            frames.append(frame)
    return frames


def get_joints(plant, model_instances=None):
    # TODO(eric.cousineau): Hoist this somewhere?
    return _get_plant_aggregate(
        plant.num_joints, plant.get_joint, JointIndex, model_instances)


def is_joint_solely_connected_to(joint, bodies):
    # TODO(eric.cousineau): Hoist this somewhere?
    parent = joint.parent_body()
    child = joint.child_body()
    return parent in bodies and child in bodies


def get_joints_solely_connected_to(plant, bodies):
    # TODO(eric.cousineau): Hoist this somewhere?
    return [
        joint for joint in get_joints(plant)
        if is_joint_solely_connected_to(joint, bodies)]


def get_joint_actuators(plant, model_instances=None):
    # TODO(eric.cousineau): Hoist this somewhere?
    return _get_plant_aggregate(
        plant.num_actuators, plant.get_joint_actuator,
        JointActuatorIndex)


def get_joint_actuators_affecting_joints(plant, joints):
    # TODO(eric.cousineau): Hoist this somewhere?
    joint_actuators = []
    for joint_actuator in get_joint_actuators(plant):
        if joint_actuator.joint() in joints:
            joint_actuators.append(joint_actuator)
    return joint_actuators


def get_or_add_model_instance(plant, name):
    # TODO(eric.cousineau): Hoist this somewhere?
    if not plant.HasModelInstanceNamed(name):
        return plant.AddModelInstance(name)
    else:
        return plant.GetModelInstanceByName(name)


def get_geometries(plant, scene_graph, bodies):
    """Returns all GeometryId's attached to bodies. Assumes corresponding
    FrameId's have been added."""
    geometry_ids = GetGeometries(plant, scene_graph, list(bodies))
    return sorted(geometry_ids, key=lambda x: x.get_value())


def get_joint_positions(plant, context, joint):
    # TODO(eric.cousineau): Hoist to C++ / pydrake.
    q = plant.GetPositions(context)
    start = joint.position_start()
    count = joint.num_positions()
    return q[start:start + count].copy()


def set_joint_positions(plant, context, joint, qj):
    # TODO(eric.cousineau): Hoist to C++ / pydrake.
    q = plant.GetPositions(context)
    start = joint.position_start()
    count = joint.num_positions()
    q[start:start + count] = qj
    plant.SetPositions(context, q)


def get_joint_velocities(plant, context, joint):
    # TODO(eric.cousineau): Hoist to C++ / pydrake.
    v = plant.GetVelocities(context)
    start = joint.velocity_start()
    count = joint.num_velocities()
    return v[start:start + count].copy()


def set_joint_velocities(plant, context, joint, vj):
    # TODO(eric.cousineau): Hoist to C++ / pydrake.
    v = plant.GetVelocities(context)
    start = joint.velocity_start()
    count = joint.num_velocities()
    v[start:start + count] = vj
    plant.SetVelocities(context, v)


def elements_sorted(xs):
    # TODO(eric.cousineau): Bind `__lt__` for sorting these types, and then
    # just use sorted().
    # Use https://github.com/RobotLocomotion/drake/pull/13489
    xs = list(xs)
    if len(xs) == 0:
        return xs
    x0 = take_first(xs)
    # TypeSafeIndex.
    try:
        int(x0)
        return sorted(xs, key=lambda x: int(x))
    except TypeError as e:
        if "int() argument" not in str(e):
            raise
    # MultibodyPlant element.
    try:
        int(x0.index())
        return sorted(xs, key=lambda x: int(x.index()))
    except AttributeError as e:
        if "has no attribute 'index'" not in str(e):
            raise
    # Geometry identifier.
    try:
        x0.get_value()
        return sorted(xs, key=lambda x: int(x.get_value()))
    except AttributeError as e:
        if "has no attribute 'get_value'" not in str(e):
            raise
    assert False


def get_frame_pose(plant, context, frame_T, frame_F):
    """Gets the pose of a frame."""
    X_TF = plant.CalcRelativeTransform(context, frame_T, frame_F)
    return X_TF


def set_frame_pose(plant, context, frame_T, frame_F, X_TF):
    """Sets the pose of a frame attached to floating body."""
    if frame_T is None:
        frame_T = plant.world_frame()
    X_WT = plant.CalcRelativeTransform(context, plant.world_frame(), frame_T)
    assert frame_F.body().is_floating()
    X_FB = frame_F.GetFixedPoseInBodyFrame().inverse()
    X_WB = X_WT @ X_TF @ X_FB
    plant.SetFreeBodyPose(context, frame_F.body(), X_WB)


def get_frame_spatial_velocity(plant, context, frame_T, frame_F, frame_E=None):
    """
    Returns:
        SpatialVelocity of frame F's origin w.r.t. frame T, expressed in E
        (which is frame T if unspecified).
    """
    if frame_E is None:
        frame_E = frame_T
    Jv_TF_E = plant.CalcJacobianSpatialVelocity(
        context,
        with_respect_to=JacobianWrtVariable.kV,
        frame_B=frame_F,
        p_BP=[0, 0, 0],
        frame_A=frame_T,
        frame_E=frame_E,
    )
    v = plant.GetVelocities(context)
    V_TF_E = SpatialVelocity(Jv_TF_E @ v)
    return V_TF_E


def set_frame_spatial_velocity(
    plant, context, frame_T, frame_F, V_TF_E, frame_E=None
):
    if frame_E is None:
        frame_E = frame_T
    R_WE = plant.CalcRelativeTransform(
        context, plant.world_frame(), frame_E
    ).rotation()
    V_TF_W = V_TF_E.Rotate(R_WE)
    X_WT = plant.CalcRelativeTransform(context, plant.world_frame(), frame_T)
    V_WT = get_frame_spatial_velocity(
        plant, context, plant.world_frame(), frame_T
    )
    V_WF = V_WT.ComposeWithMovingFrameVelocity(X_WT.translation(), V_TF_W)
    body_B = frame_F.body()
    R_WB = plant.CalcRelativeTransform(
        context, plant.world_frame(), body_B.body_frame()
    ).rotation()
    p_BF = frame_F.GetFixedPoseInBodyFrame().translation()
    p_BF_W = R_WB @ p_BF
    V_WBf = V_WF.Shift(p_BF_W)
    plant.SetFreeBodySpatialVelocity(body_B, V_WBf, context)


def remove_role_from_geometries(plant, scene_graph, *, role, bodies=None):
    if bodies is None:
        bodies = get_bodies(plant)
    return RemoveRoleFromGeometries(plant, scene_graph, role, list(bodies))
