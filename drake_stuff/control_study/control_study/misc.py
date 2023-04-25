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

from control_study.spaces import (
    declare_spatial_motion_outputs,
)
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


class PoseTraj(LeafSystem):
    def __init__(self, traj):
        super().__init__()
        self.traj = traj

        def calc_cache(context, cache):
            t = context.get_time()
            cache.X, cache.V, cache.A = self.traj(t)

        cache_entry = declare_simple_cache(self, calc_cache)
        self.motion_outputs = declare_spatial_motion_outputs(
            self,
            frames=None,
            name_X="X",
            calc_X=lambda context, output: (
                output.set_value(cache_entry.Eval(context).X)
            ),
            name_V="V",
            calc_V=lambda context, output: (
                output.set_value(cache_entry.Eval(context).V)
            ),
            name_A="A",
            calc_A=lambda context, output: (
                output.set_value(cache_entry.Eval(context).A)
            ),
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
    parser.AddModels(
        "package://drake/manipulation/models/franka_description/urdf/panda_arm.urdf"
    )
    plant.Finalize()
    plant.WeldFrames(
        plant.world_frame(),
        plant.GetFrameByName("panda_link0"),
    )
    frame_G = plant.GetFrameByName("panda_link8")

    return builder, plant, scene_graph, frame_G


def xyz_rpy(xyz, rpy):
    """Shorthand to create an isometry from XYZ and RPY."""
    return RigidTransform(R=RotationMatrix(rpy=RollPitchYaw(rpy)), p=xyz)


def xyz_rpy_deg(xyz, rpy_deg):
    return xyz_rpy(xyz, np.deg2rad(rpy_deg))
