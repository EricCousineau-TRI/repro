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
from pydrake.multibody.tree import ModelInstanceIndex
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder, LeafSystem

from control_study.multibody_extras import (
    get_frame_spatial_velocity,
    simplify_plant,
)
from control_study.spaces import (
    connect_spatial_motion_ports,
    declare_spatial_motion_outputs,
)
from control_study.systems import (
    declare_simple_cache,
    maybe_attach_zoh,
    simulator_initialize_repeatedly,
)
from control_study.trajectories import (
    make_se3_spline_trajectory,
)
from control_study.misc import (
    make_sim_setup,
    PoseTraj,
    xyz_rpy_deg,
    SimpleLogger,
    make_sample_pose_traj,
    unzip,
)

CONTROL_DT = 0.002
DISCRETE_PLANT_TIME_STEP = 0.0008


def run_spatial_waypoints(
    *,
    make_controller,
    X_extr,
    X_intr,
    dT,
    discrete=True,
):
    if discrete:
        control_dt = CONTROL_DT
        plant_time_step = DISCRETE_PLANT_TIME_STEP
    else:
        control_dt = None
        plant_time_step = 0.0
    control_dt = CONTROL_DT

    builder, plant, scene_graph, frame_G = make_sim_setup(plant_time_step)
    frame_W = plant.world_frame()

    controller = builder.AddSystem(
        make_controller(plant, frame_W, frame_G)
    )
    torques_output = maybe_attach_zoh(controller.torques_output, control_dt)
    # Meh
    model = ModelInstanceIndex(2)
    builder.Connect(
        torques_output,
        plant.get_actuation_input_port(model),
    )

    # Simplify plant after controller is constructed.
    simplify_plant(plant, scene_graph)

    traj_sys = builder.AddSystem(PoseTraj(traj=None))
    connect_spatial_motion_ports(
        builder,
        outputs=traj_sys.motion_outputs,
        inputs=controller.inputs_motion_desired,
    )

    def log_instant(log_context):
        q = plant.GetPositions(context)
        V = get_frame_spatial_velocity(
            plant, context, frame_W, frame_G
        )
        V = V.get_coeffs().copy()
        return q, V

    logger = builder.AddSystem(
        SimpleLogger(period_sec=0.01, func=log_instant)
    )

    diagram = builder.Build()
    diagram_context = diagram.CreateDefaultContext()
    context = plant.GetMyContextFromRoot(diagram_context)

    # Simple bent-elbow and wrist-down pose.
    q0 = np.deg2rad([0.0, 15.0, 0.0, -75.0, 0.0, 90.0, 0.0])
    plant.SetPositions(context, q0)

    X_WG = plant.CalcRelativeTransform(context, frame_W, frame_G)
    traj_saturate, t_f = make_sample_pose_traj(dT, X_WG, X_extr, X_intr)

    traj_sys.traj = traj_saturate

    simulator = Simulator(diagram, diagram_context)
    simulator_initialize_repeatedly(simulator)

    try:
        # Run a bit past the end of trajectory.
        simulator.AdvanceTo(t_f + dT)
        simulator.AdvancePendingEvents()
    except Exception:
        print(f"Failure at {diagram_context.get_time()}s")
        raise

    # Return logged values.
    qs, Vs = unzip(logger.log)
    qs, Vs = map(np.array, (qs, Vs))
    return qs, Vs


def make_controller(plant):
    pass


def run_rotation_coupling():
    # Rotation only.
    run_spatial_waypoints(
        X_extr=RigidTransform([0, 0, 0]),
        X_intr=xyz_rpy_deg([0, 0, 0], [0.0, 0.0, 175.0]),
        dT=1.0,
    )


def run_slow_waypoints():
    # Small and slow(er) motion, should stay away from singularities and
    # stay within limits of QP, so all should be (relatively) equivalent.
    run_spatial_waypoints(
        X_extr=RigidTransform([0.05, 0.02, -0.05]),
        X_intr=xyz_rpy_deg([0, 0, 0], [90.0, 30.0, 45.0]),
        dT=2.0,
    )


def run_fast_waypoints_singular():
    # Fast motions that move beyond our speed limits and move into
    # singularity (elbow lock).
    run_spatial_waypoints(
        X_extr=RigidTransform([0.5, 0.2, -0.5]),
        X_intr=xyz_rpy_deg([0, 0, 0], [90.0, 30.0, 45.0]),
        dT=1.0,
    )
