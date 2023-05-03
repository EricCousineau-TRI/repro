import numpy as np

from pydrake.geometry import CollisionFilterDeclaration, GeometrySet
from pydrake.multibody.math import SpatialVelocity
from pydrake.multibody.tree import (
    BodyIndex,
    FrameIndex,
    JacobianWrtVariable,
    JointActuatorIndex,
    JointIndex,
    ModelInstanceIndex,
    PrismaticJoint,
    RevoluteJoint,
)

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


def get_geometries(plant, scene_graph, bodies):
    """Returns all GeometryId's attached to bodies. Assumes corresponding
    FrameId's have been added."""
    geometry_ids = []
    inspector = scene_graph.model_inspector()
    for geometry_id in inspector.GetAllGeometryIds():
        body = plant.GetBodyFromFrameId(inspector.GetFrameId(geometry_id))
        if body in bodies:
            geometry_ids.append(geometry_id)
    return sorted(geometry_ids, key=lambda x: x.get_value())


def filter_all_collisions(plant, scene_graph):
    bodies = get_bodies(plant)
    geometries = get_geometries(plant, scene_graph, bodies)
    filter_manager = scene_graph.collision_filter_manager()
    geometry_set = GeometrySet(geometries)
    declaration = CollisionFilterDeclaration()
    declaration.ExcludeWithin(geometry_set)
    filter_manager.Apply(declaration)


def remove_joint_limits(plant):
    # TODO(eric.cousineau): Handle actuator limits when Drake supports mutating
    # them.
    for joint in get_joints(plant):
        num_q = joint.num_positions()
        num_v = joint.num_velocities()
        joint.set_position_limits(
            np.full(num_q, -np.inf), np.full(num_q, np.inf)
        )
        joint.set_velocity_limits(
            np.full(num_v, -np.inf), np.full(num_v, np.inf)
        )
        joint.set_acceleration_limits(
            np.full(num_v, -np.inf), np.full(num_v, np.inf)
        )


def set_zero_gravity(plant):
    plant.mutable_gravity_field().set_gravity_vector(np.zeros(3))


def simplify_plant(plant, scene_graph, *, zero_gravity=True):
    """
    Zeros out gravity, removes collisions, and effectively disables joint
    limits.
    """
    if zero_gravity:
        set_zero_gravity(plant)
    filter_all_collisions(plant, scene_graph)
    remove_joint_limits(plant)


def calc_velocity_jacobian(
    plant,
    context,
    frame_W,
    frame_B,
    *,
    frame_E=None,
    p_BoBp_B=None,
    include_bias=False,
):
    if frame_E is None:
        frame_E = frame_W
    if p_BoBp_B is None:
        p_BoBp_B = np.zeros(3)
    J = plant.CalcJacobianSpatialVelocity(
        context,
        with_respect_to=JacobianWrtVariable.kV,
        frame_B=frame_B,
        p_BoBp_B=p_BoBp_B,
        frame_A=frame_W,
        frame_E=frame_E,
    )
    if include_bias:
        Jd_v = plant.CalcBiasSpatialAcceleration(
            context,
            with_respect_to=JacobianWrtVariable.kV,
            frame_B=frame_B,
            p_BoBp_B=p_BoBp_B,
            frame_A=frame_W,
            frame_E=frame_E,
        ).get_coeffs()
        return J, Jd_v
    else:
        return J


def get_frame_spatial_velocity(plant, context, frame_T, frame_F, frame_E=None):
    """
    Returns:
        SpatialVelocity of frame F's origin w.r.t. frame T, expressed in E
        (which is frame T if unspecified).
    """
    Jv_TF_E = calc_velocity_jacobian(
        plant, context, frame_T, frame_F, frame_E=frame_E
    )
    v = plant.GetVelocities(context)
    V_TF_E = SpatialVelocity(Jv_TF_E @ v)
    return V_TF_E
