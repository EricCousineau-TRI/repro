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
from pydrake.systems.analysis import (
    ApplySimulatorConfig,
    Simulator,
    SimulatorConfig,
)
from pydrake.systems.framework import DiagramBuilder, LeafSystem

from control_study import debug
from control_study.controllers import (
    Osc,
    OscGains,
    QpWithCosts,
    QpWithDirConstraint,
)
from control_study.limits import PlantLimits
from control_study.multibody_extras import (
    get_frame_spatial_velocity,
    simplify_plant,
)
from control_study.systems import (
    declare_simple_cache,
    maybe_attach_zoh,
    simulator_initialize_repeatedly,
)
from control_study.trajectories import (
    make_se3_spline_trajectory,
)
from control_study.geometry import xyz_rpy_deg
from control_study.misc import (
    make_sim_setup,
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
    # if discrete:
        # control_dt = CONTROL_DT
        # plant_time_step = DISCRETE_PLANT_TIME_STEP
    # else:
        # control_dt = None
        # plant_time_step = 0.0
    plant_time_step = 0.0
    # control_dt = CONTROL_DT
    control_dt = None

    builder, plant, scene_graph, frame_G = make_sim_setup(plant_time_step)
    frame_W = plant.world_frame()

    controller = builder.AddSystem(
        make_controller(plant, frame_W, frame_G)
    )
    builder.Connect(
        plant.get_state_output_port(),
        controller.state_input,
    )
    torques_output = maybe_attach_zoh(
        builder, controller.torques_output, control_dt
    )
    # Meh, should fix this,
    model = ModelInstanceIndex(2)
    builder.Connect(
        torques_output,
        plant.get_actuation_input_port(model),
    )

    # Simplify plant after controller is constructed.
    simplify_plant(plant, scene_graph)

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
    controller.traj = traj_saturate

    simulator = Simulator(diagram, diagram_context)
    config = SimulatorConfig(
        integration_scheme="explicit_euler",
        max_step_size=CONTROL_DT,
    )
    ApplySimulatorConfig(config, simulator)
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


def run_rotation_coupling(make_controller):
    # Rotation only.
    run_spatial_waypoints(
        make_controller=make_controller,
        X_extr=RigidTransform([0, 0, 0]),
        X_intr=xyz_rpy_deg([0, 0, 0], [0.0, 0.0, 175.0]),
        dT=1.0,
    )


def run_slow_waypoints(make_controller):
    # Small and slow(er) motion, should stay away from singularities and
    # stay within limits of QP, so all should be (relatively) equivalent.
    run_spatial_waypoints(
        make_controller=make_controller,
        X_extr=RigidTransform([0.05, 0.02, -0.05]),
        X_intr=xyz_rpy_deg([0, 0, 0], [90.0, 30.0, 45.0]),
        dT=2.0,
    )


def run_fast_waypoints(make_controller):
    # Small and slow(er) motion, should stay away from singularities and
    # stay within limits of QP, so all should be (relatively) equivalent.
    run_spatial_waypoints(
        make_controller=make_controller,
        X_extr=RigidTransform([0.05, 0.02, -0.05]),
        X_intr=xyz_rpy_deg([0, 0, 0], [90.0, 30.0, 45.0]),
        dT=0.5,
    )

def run_fast_waypoints_singular(make_controller):
    # Fast motions that move beyond our speed limits and move into
    # singularity (elbow lock).
    run_spatial_waypoints(
        make_controller=make_controller,
        X_extr=RigidTransform([0.5, 0.2, -0.5]),
        X_intr=xyz_rpy_deg([0, 0, 0], [90.0, 30.0, 45.0]),
        dT=1.0,
    )


def make_osc_gains():
    return OscGains.critically_damped(10.0, 10.0)  # No elbow sticking
    # return OscGains.critically_damped(10.0, 1.0)  # No elbow sticking
    # return OscGains.critically_damped(100.0, 100.0)  # No elbow sticking, but loses tracking
    # return OscGains.critically_damped(100.0, 10.0)  # Stickier
    # return OscGains.critically_damped(100.0, 1.0)  # Drifts... hard...
    # return OscGains.critically_damped(100.0, 0.0)  # of course locks


def make_panda_limits(plant):
    plant_limits = PlantLimits.from_plant(plant)
    # Avoid elbow lock.
    plant_limits.q = plant_limits.q.scaled(0.95)
    plant_limits.q.upper[3] = np.deg2rad(-20.0)  # Locks...
    # plant_limits.q.upper[3] = np.deg2rad(-25.0)  # Does not lock...
    plant_limits.v = plant_limits.v.scaled(0.95)
    # plant_limits.v = plant_limits.v.scaled(0.9)
    # plant_limits.vd = plant_limits.vd.scaled(0.95)  # causes issues
    # plant_limits.u = plant_limits.u.scaled(0.99)  # causes issues
    # plant_limits.v = plant_limits.v.scaled(0.5)
    # plant_limits.vd = plant_limits.vd.scaled(np.inf)
    # plant_limits.u = plant_limits.u.scaled(0.95)
    # plant_limits.v = plant_limits.vd.scaled(np.inf)
    # plant_limits.vd = plant_limits.vd.scaled(np.inf)
    # plant_limits.u = plant_limits.u.scaled(np.inf)
    return plant_limits


def make_controller_osc(plant, frame_W, frame_G):
    # Great at tracking
    # Bad at singularities or staying w/in bounds.
    # TODO(eric.cousineau): Should try out potential fields.
    controller = Osc(
        plant,
        frame_W,
        frame_G,
        gains=make_osc_gains(),
    )
    controller.check_limits = False
    return controller


def make_controller_qp_costs(plant, frame_W, frame_G):
    # Good with singularity and speed.
    # Bad with rotation / maintaining direction.
    return QpWithCosts(
        plant,
        frame_W,
        frame_G,
        gains=make_osc_gains(),
        plant_limits=make_panda_limits(plant),
        acceleration_bounds_dt=CONTROL_DT,
        posture_weight=0.1,
        # split_costs=[1.0, 10.0],
        split_costs=None,
        # posture_weight=1.0,
        # use_torque_weights=True,
    )


def make_controller_qp_constraints(plant, frame_W, frame_G):
    # at present, elbow oscillates at limit; could either be secondary task, or
    # something else.
    controller = QpWithDirConstraint(
        plant,
        frame_W,
        frame_G,
        gains=make_osc_gains(),
        plant_limits=make_panda_limits(plant),
        acceleration_bounds_dt=100 * CONTROL_DT,
        # acceleration_bounds_dt=5 * CONTROL_DT,
        # posture_weight=0.01,
        # use_torque_weights=False,
        posture_weight=1.0,
        use_torque_weights=True,
    )
    # controller.check_limits = False
    return controller


def np_print_more_like_matlab():
    np.set_printoptions(
        formatter={"float_kind": lambda x: f"{x: 06.3f}"},
        linewidth=150,
    )


@debug.iex
def main():
    np_print_more_like_matlab()
    scenarios = {
        # "slow": run_slow_waypoints,
        # "rot": run_rotation_coupling,
        # "fast": run_fast_waypoints,
        "fast singular": run_fast_waypoints_singular,
    }
    make_controllers = {
        # "osc": make_controller_osc,
        # "qp costs": make_controller_qp_costs,
        "qp constr": make_controller_qp_constraints,
    }
    for scenario_name, scenario in scenarios.items():
        print(scenario_name)
        for controller_name, make_controller in make_controllers.items():
            print(f"  {controller_name}")
            try:
                scenario(make_controller)
            except RuntimeError as e:
                print(e)
                raise  # wip


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass