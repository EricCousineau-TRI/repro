import numpy as np

from pydrake.geometry import DrakeVisualizer, DrakeVisualizerParams, Role
from pydrake.math import RigidTransform, RollPitchYaw, RotationMatrix
from pydrake.multibody.math import (
    SpatialAcceleration,
    SpatialForce,
    SpatialVelocity,
)
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlant, MultibodyPlantConfig
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder, LeafSystem

from control_study.systems import (
    declare_simple_cache,
)
from control_study.trajectories import (
    make_se3_spline_trajectory,
)

def unzip(xs):
    return tuple(zip(*xs, strict=True))


class SimpleLogger(LeafSystem):
    def __init__(self, period_sec, func):
        super().__init__()
        # WARNING: This is undeclared state! This will only work for a single
        # simulation.
        self.log = []

        def on_discrete_update(context, raw_state):
            instant = func(context)
            self.log.append(instant)

        self.DeclarePeriodicUnrestrictedUpdateEvent(
            period_sec=period_sec,
            offset_sec=0.0,
            update=on_discrete_update,
        )


def pad_tape(ts, Xs, pad):
    ts = np.array(ts)
    ts += pad
    ts = [0.0] + ts.tolist() + [ts[-1] + pad]
    ts = np.asarray(ts)
    Xs = [Xs[0]] + Xs + [Xs[-1]]
    return ts, Xs


def make_sample_pose_traj(dT, X_WG, X_extr, X_intr):
    Xs = [
        X_WG,
        X_extr @ X_WG @ X_intr,
        X_WG,
        X_extr.inverse() @ X_WG @ X_intr.inverse(),
        X_WG,
    ]
    ts = np.arange(len(Xs)) * dT
    ts, Xs = pad_tape(ts, Xs, pad=dT / 4.0)
    traj = make_se3_spline_trajectory(ts, Xs)
    t_f = ts[-1]

    def traj_saturate(t):
        # Saturate at end.
        if t >= t_f:
            t = t_f
        X, V, A = traj(t)
        return X, V, A

    return traj_saturate, t_f


def make_sim_setup(time_step):
    builder = DiagramBuilder()
    config = MultibodyPlantConfig(
        time_step=time_step,
        discrete_contact_solver="sap",
    )
    plant, scene_graph = AddMultibodyPlant(config, builder)
    DrakeVisualizer.AddToBuilder(
        builder,
        scene_graph,
        params=DrakeVisualizerParams(role=Role.kIllustration),
    )
    parser = Parser(plant)
    parser.AddModelsFromUrl(
        "package://drake/manipulation/models/franka_description/urdf/panda_arm.urdf"
    )
    plant.WeldFrames(
        plant.world_frame(),
        plant.GetFrameByName("panda_link0"),
    )
    plant.Finalize()

    frame_G = plant.GetFrameByName("panda_link8")
    return builder, plant, scene_graph, frame_G
