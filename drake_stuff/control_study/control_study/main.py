from functools import partial

import numpy as np

from pydrake.math import RigidTransform
from pydrake.systems.analysis import (
    ApplySimulatorConfig,
    Simulator,
    SimulatorConfig,
)
from pydrake.systems.framework import DiagramBuilder, LeafSystem

from control_study import debug
import control_study.controllers as m
from control_study.controllers import (
    DiffIkAndId,
    Gains,
    Osc,
    OscGains,
    QpWithCosts,
    QpWithDirConstraint,
    ResolvedAcc,
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
    DirectPlant,
    EulerAccelPlant,
    SimpleLogger,
    load_pickle,
    make_sample_pose_traj,
    make_sim_setup,
    np_print_more_like_matlab,
    unzip,
)

CONTROL_DT = 0.002
# CONTROL_DT = 0.01
DISCRETE_PLANT_TIME_STEP = 0.0008


def run_control(
    *,
    make_controller,
    make_traj,
    q0,
    X_WB=RigidTransform(),
):
    plant_time_step = DISCRETE_PLANT_TIME_STEP
    control_dt = CONTROL_DT
    # plant_time_step = 0.0
    # control_dt = None

    plant_diagram, plant, scene_graph, frame_G = make_sim_setup(
        plant_time_step, X_WB
    )
    frame_W = plant.world_frame()

    builder = DiagramBuilder()
    access = DirectPlant.AddToBuilder(builder, plant_diagram, plant)
    # access = EulerAccelPlant.AddToBuilder(
    #     builder, plant_diagram, plant, CONTROL_DT
    # )

    controller = builder.AddSystem(
        make_controller(plant, frame_W, frame_G)
    )
    builder.Connect(
        access.state_output_port,
        controller.state_input,
    )
    torques_output = maybe_attach_zoh(
        builder, controller.torques_output, control_dt
    )
    builder.Connect(
        torques_output,
        access.torque_input_port,
    )

    # Simplify plant after controller is constructed.
    simplify_plant(plant, scene_graph)

    def log_instant(log_context):
        context = access.read_plant_context(diagram_context)
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
    context = access.read_plant_context(diagram_context)

    plant.SetPositions(context, q0)

    X_WG = plant.CalcRelativeTransform(context, frame_W, frame_G)
    traj, t_f = make_traj(X_WG)
    controller.traj = traj

    access.write_plant_context(diagram_context, context)

    simulator = Simulator(diagram, diagram_context)
    # config = SimulatorConfig(
    #     integration_scheme="explicit_euler",
    #     max_step_size=CONTROL_DT,
    # )
    # ApplySimulatorConfig(config, simulator)
    simulator_initialize_repeatedly(simulator)
    simulator.set_target_realtime_rate(1.0)

    def continuous_monitor(_):
        controller.should_save = True
        assert torques_output.get_system() is controller
        context = controller.GetMyContextFromRoot(diagram_context)
        torques_output.Eval(context)
        controller.should_save = False

    if control_dt is None:
        simulator.set_monitor(continuous_monitor)
    else:
        # ZOH means each step is actual step.
        controller.should_save = True

    m.SHOULD_STOP = True
    do_plot = False

    try:
        # Run a bit past the end of trajectory.
        simulator.AdvanceTo(t_f)
        # simulator.AdvanceTo(1.25)
        # simulator.AdvanceTo(1.5)  # HACK
        simulator.AdvancePendingEvents()
    except (Exception, KeyboardInterrupt) as e:
        err_name = type(e).__name__
        print(f"{err_name} at {diagram_context.get_time()}s")
        if isinstance(e, (RuntimeError, KeyboardInterrupt)):
            if do_plot:
                controller.show_plots()
            pass
        raise

    if do_plot:
        controller.show_plots()

    # Return logged values.
    qs, Vs = unzip(logger.log)
    qs, Vs = map(np.array, (qs, Vs))
    return qs, Vs


def run_spatial_waypoints(
    *,
    make_controller,
    X_extr,
    X_intr,
    dT,
):
    def make_traj(X_WG):
        traj_saturate, t_f = make_sample_pose_traj(dT, X_WG, X_extr, X_intr)
        t_f += dT
        return traj_saturate, t_f

    # Simple bent-elbow and wrist-down pose.
    q0 = np.deg2rad([0.0, 15.0, 0.0, -75.0, 0.0, 90.0, 0.0])
    return run_control(
        make_controller=make_controller,
        make_traj=make_traj,
        q0=q0,
    )


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

def run_fast_waypoints_singular(make_controller, *, rotate):
    # Fast motions that move beyond our speed limits and move into
    # singularity (elbow lock).
    if rotate:
        X_intr = xyz_rpy_deg([0, 0, 0], [90.0, 30.0, 45.0])
    else:
        X_intr = RigidTransform()
    run_spatial_waypoints(
        make_controller=make_controller,
        X_extr=RigidTransform([0.5, 0.2, -0.5]),
        X_intr=X_intr,
        dT=1.0,
    )


def make_osc_gains():
    # return OscGains.critically_damped(10.0, 0.0)  # like diff ik
    return OscGains(
        task=Gains.critically_damped(10.0),
        # stability is noted for teleop trajectory case...
        posture=Gains(kp=100.0, kd=20.0),  # unstable (towards end)
        # posture=Gains(kp=0.0, kd=20.0),  # stable
        # posture=Gains(kp=-100.0, kd=20.0),  # stable?!!!
        # posture=Gains(kp=-100.0, kd=-20.0),  # unstable
        # posture=Gains(kp=0.0, kd=-20.0),  # unstable
        # posture=Gains(kp=100.0, kd=-20.0),  # unstable
    )
    # return OscGains.critically_damped(10.0, 100.0)  # goes crazy

    # return OscGains.critically_damped(1.0, 1.0)
    # return OscGains.critically_damped(10.0, 10.0)
    # return OscGains.critically_damped(10.0, 1.0)
    # return OscGains.critically_damped(100.0, 100.0)
    # return OscGains.critically_damped(100.0, 10.0)
    # return OscGains.critically_damped(150.0, 15.0)
    # return OscGains.critically_damped(300.0, 30.0)
    # return OscGains.critically_damped(100.0, 1.0)  # Drifts... hard...
    # return OscGains.critically_damped(100.0, 0.0)  # of course locks


def make_panda_limits(plant):
    plant_limits = PlantLimits.from_plant(plant)
    # Avoid elbow lock.
    # plant_limits.q = plant_limits.q.scaled(0.9)
    # plant_limits.v = plant_limits.v.scaled(0.9)
    # plant_limits.q.upper[3] = np.deg2rad(-5.0)
    # plant_limits.q.upper[3] = np.deg2rad(-10.0)
    # plant_limits.q.upper[3] = np.deg2rad(-15.0)
    # plant_limits.q.upper[3] = np.deg2rad(-20.0)  # vibrates, locks
    # plant_limits.q.upper[3] = np.deg2rad(-25.0)  # vibrates
    # plant_limits.q.upper[3] = np.deg2rad(-30.0)
    # plant_limits.q.upper[3] = np.deg2rad(-35.0)  # near singular value=0.01
    # plant_limits.q.upper[3] = np.deg2rad(-45.0)
    # plant_limits.q.lower[6] = np.deg2rad(-30.0)
    # plant_limits.q.upper[6] = np.deg2rad(30.0)
    plant_limits.v = plant_limits.v.scaled(0.95)
    # plant_limits.vd = plant_limits.vd.scaled(0.95)  # causes issues
    # plant_limits.u = plant_limits.u.scaled(0.99)  # causes issues
    # plant_limits.v = plant_limits.v.scaled(0.5)
    # plant_limits.vd = plant_limits.vd.scaled(np.inf)
    # plant_limits.u = plant_limits.u.scaled(0.95)
    # plant_limits.q = plant_limits.q.scaled(np.inf)
    # plant_limits.v = plant_limits.v.scaled(np.inf)
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


def make_controller_resolved_acc(plant, frame_W, frame_G):
    controller = ResolvedAcc(
        plant,
        frame_W,
        frame_G,
        gains=make_osc_gains(),
    )
    controller.check_limits = False
    return controller


def make_controller_diff_ik(plant, frame_W, frame_G):
    controller = DiffIkAndId(
        plant,
        frame_W,
        frame_G,
        dt=CONTROL_DT,
        # A bit high, but eh.
        gains_p=Gains.critically_damped(100.0),
    )
    controller.check_limits = True
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
        acceleration_bounds_dt=CONTROL_DT,
        # acceleration_bounds_dt=100 * CONTROL_DT,
        # acceleration_bounds_dt=2 * CONTROL_DT,
        # posture_weight=0.05,
        # use_torque_weights=False,
        posture_weight=1.0,
        use_torque_weights=True,
    )
    # controller.check_limits = False
    return controller


def load_teleop_traj(*, as_spline=True):
    data = load_pickle("./data/osc_wrap_sim_panda.pkl")
    tape = data.tape
    q0 = data.q0
    # Offset from pose of object in original file, to be removed from traj.
    p_WB = np.array([-0.75, 0.0, -0.2])
    for item in tape:
        item.X_des = RigidTransform(-p_WB) @ item.X_des

    if as_spline:
        ts = np.array([item.t for item in tape])
        Xs = np.array([item.X_des for item in tape])
        downsample = slice(None, None, 5)
        ts = ts[downsample]
        Xs = Xs[downsample]
        traj_raw = make_se3_spline_trajectory(ts, Xs)
        t_f_actual = ts[-1]

        def traj(t):
            if t > t_f_actual:
                t = t_f_actual
            return traj_raw(t)

        t_f = t_f_actual * 1.2
    else:
        t_f = tape[-1].t
        # Only for discretized control!
        i = 0

        def traj(t):
            nonlocal i
            if t > tape[i].t:
                i += 1
            item = tape[i]
            assert item.t == t
            return item.X_des, item.V_des, item.A_des

    return q0, traj, t_f


def run_teleop_traj(make_controller):
    q0, traj, t_f = load_teleop_traj(as_spline=True)
    run_control(
        make_controller=make_controller,
        make_traj=lambda X_WG: (traj, t_f),
        q0=q0,
    )


def scenario_main():
    scenarios = {
        # "slow": run_slow_waypoints,
        "rot": run_rotation_coupling,
        # "fast": run_fast_waypoints,
        # "teleop": run_teleop_traj,
        # "fast singular": partial(run_fast_waypoints_singular, rotate=False),
        # "fast singular rot": partial(run_fast_waypoints_singular, rotate=True),
    }
    make_controllers = {
        "diff ik": make_controller_diff_ik,
        # "acc": make_controller_resolved_acc,
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
            except (RuntimeError, AssertionError, KeyboardInterrupt) as e:
                print(e)
                # raise  # wip


@debug.iex
def main():
    np_print_more_like_matlab()
    scenario_main()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
