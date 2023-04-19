"""
Tests basic stiction simulation.

Goal:
- Have simple, single-rotor setup. Just lop off last link of Panda (J7).
- Get simple velocity control working.
- Add in simple friction model.
- Try to reproduce (v, u) measurements.
"""

import copy
from functools import partial
from types import SimpleNamespace
import unittest

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import tqdm

from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import (
    AddMultibodyPlant,
    MultibodyPlant,
    MultibodyPlantConfig,
)
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder, LeafSystem
from pydrake.systems.primitives import Adder
from pydrake.visualization import ApplyVisualizationConfig, VisualizationConfig

from anzu.not_exported_soz import dict_items_zip
from anzu.not_exported_soz import parallel_work
from anzu.friction import (
    calc_joint_dry_friction,
    regularizer_arctan,
    regularizer_tanh,
)
from anzu.friction_fit import (
    JointDryFriction,
    JointFriction,
    RangeConstrainedScalar,
    param_value,
    plot_fit_experiment,
    run_fit_experiment,
)
from anzu.not_exported_soz import maybe_attach_zoh
from anzu.not_exported_soz import cat, maxabs
from anzu.not_exported_soz import resolve_path

VISUALIZE = False


class Logger(LeafSystem):
    def __init__(self, *, dt):
        super().__init__()
        self.x = self.DeclareVectorInputPort("x", 2)
        self.u = self.DeclareVectorInputPort("u", 1)

        # WARNING: This is undeclared state! This will only work for a single
        # simulation.
        log = []

        def eval_log(context):
            return log

        def on_discrete_update(context, raw_state):
            t = context.get_time()
            x = self.x.Eval(context)
            u = self.u.Eval(context)
            q = x[:1]
            v = x[1:]
            instant = SimpleNamespace(t=t, q=q, v=v, u=u)
            log.append(instant)

        self.DeclarePeriodicUnrestrictedUpdateEvent(
            period_sec=dt,
            offset_sec=0.0,
            update=on_discrete_update,
        )
        self.eval_log = eval_log


class StateToTorque(LeafSystem):
    def __init__(self, calc_u):
        super().__init__()
        self.x = self.DeclareVectorInputPort("x", 2)

        def calc_u_output(context, output):
            x = self.x.Eval(context)
            t = context.get_time()
            q = x[:1]
            v = x[1:]
            u = calc_u(t, q, v)
            output.set_value(u)

        self.u = self.DeclareVectorOutputPort("u", 1, calc_u_output)


class Noise(LeafSystem):
    def __init__(self, amp):
        super().__init__()
        N = len(amp)

        def calc_y(context, output):
            y = amp * np.random.uniform(low=-1.0, high=1.0, size=N)
            output.set_value(y)

        self.y = self.DeclareVectorOutputPort("y", N, calc_y)


def add_noise(builder, output_port, amp, zoh_dt):
    N = len(amp)
    noise = builder.AddSystem(Noise(amp))
    adder = builder.AddSystem(Adder(num_inputs=2, size=N))
    noise_x = maybe_attach_zoh(builder, noise.get_output_port(), zoh_dt)
    builder.Connect(output_port, adder.get_input_port(0))
    builder.Connect(noise_x, adder.get_input_port(1))
    return adder.get_output_port()


def add_model(plant):
    model_file = resolve_path("package://anzu/models/haptic/panda_j7.urdf")
    model = Parser(plant).AddModelFromFile(model_file)
    # Posture the link with rotary axis aligned with gravity.
    plant.WeldFrames(
        plant.world_frame(),
        plant.GetFrameByName("panda_joint7_inboard"),
    )
    return model


def run_friction_sim(
    calc_u, calc_friction, *, noise_amp=None, tf=1.0, dt=None, visualize=True
):
    builder = DiagramBuilder()
    config = MultibodyPlantConfig(time_step=0.0)
    plant, scene_graph = AddMultibodyPlant(config, builder)
    model = add_model(plant)
    plant.Finalize()
    if visualize:
        viz_config = VisualizationConfig()
        ApplyVisualizationConfig(
            viz_config, builder, plant=plant, scene_graph=scene_graph
        )

    u_adder = builder.AddSystem(Adder(num_inputs=2, size=1))

    controller_x = plant.get_state_output_port()
    if noise_amp is not None:
        controller_x = add_noise(
            builder, controller_x, noise_amp, zoh_dt=0.001
        )

    controller = builder.AddSystem(StateToTorque(calc_u))
    controller_u = maybe_attach_zoh(builder, controller.u, dt)
    builder.Connect(controller_x, controller.x)
    builder.Connect(controller_u, u_adder.get_input_port(0))

    stiction = builder.AddSystem(StateToTorque(calc_friction))
    builder.Connect(plant.get_state_output_port(), stiction.x)
    builder.Connect(stiction.u, u_adder.get_input_port(1))

    u_sum = u_adder.get_output_port()
    builder.Connect(u_sum, plant.get_actuation_input_port(model))

    log_dt = 0.001
    logger = builder.AddSystem(Logger(dt=log_dt))
    builder.Connect(plant.get_state_output_port(), logger.x)
    builder.Connect(controller_u, logger.u)

    diagram = builder.Build()
    diagram_context = diagram.CreateDefaultContext()
    simulator = Simulator(diagram, diagram_context)
    simulator.AdvanceTo(tf)

    logger_context = logger.GetMyContextFromRoot(diagram_context)
    log = logger.eval_log(logger_context)
    return log


def extract_log_values(log):
    """Extracts values from Logger."""
    ts = []
    qs = []
    vs = []
    us = []
    for instant in log:
        ts.append(instant.t)
        qs.append(instant.q)
        vs.append(instant.v)
        us.append(instant.u)
    (ts, qs, vs, us) = map(np.asarray, (ts, qs, vs, us))
    return (ts, qs, vs, us)


def calc_mean_velocity_and_torque(ts, vs, us):
    """Computes a reasonable mean velocity and torque from log data."""
    # Take last half (assuming steady state)
    mask = ts > (ts[-1] / 2)
    vm = np.mean(vs[mask])
    um = np.mean(us[mask])
    return vm, um


def calc_friction_sample(t, q, v, *, v0=0.05, m=0.8, b=0.0):
    regularizer = partial(regularizer_tanh, m=m)
    u_dry = calc_joint_dry_friction(
        v, v0=v0, regularizer=regularizer, u_max=0.4
    )
    u_viscous = -b * v
    return u_dry + u_viscous


def no_friction(t, q, v):
    return np.zeros(1)


def calc_u_vel(t, q, v, kd, v_d):
    v_err = v - v_d
    u = -kd * v_err
    return u


def run_friction_experiments(calc_friction):
    def worker(v_ds):
        for v_d in v_ds:
            log = run_friction_sim(
                calc_u=partial(calc_u_vel, kd=1.0, v_d=v_d),
                calc_friction=calc_friction,
                tf=1.0,
                visualize=False,
            )
            ts, qs, vs, us = extract_log_values(log)
            vm, um = calc_mean_velocity_and_torque(ts, vs, us)
            yield (vm, um)

    vs_d_abs = np.array([0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0])
    vs_d = cat(-vs_d_abs[::-1], vs_d_abs)
    # The slow velocites are stiffer to (naively) simulate.
    pairs = parallel_work(
        worker,
        vs_d,
        process_count=5,
        progress_cls=tqdm.tqdm,
    )
    vs_m, us_m = np.array(pairs).T
    return (vs_m, us_m)


def get_rotational_axis_outboard_inertia(plant, joint):
    """
    Gets the scalar rotational inertia Izz around the z-axis of a given joint.
    """
    indexes = plant.GetBodiesKinematicallyAffectedBy([joint.index()])
    context = plant.CreateDefaultContext()
    M = plant.CalcSpatialInertia(context, joint.frame_on_child(), indexes)
    Ir = M.CalcRotationalInertia()
    assert np.all(joint.revolute_axis() == [0, 0, 1])
    Izz = Ir[2, 2]
    return Izz


def run_compensation_example(*, v_d, v_noise):
    """
    With a given desired velocity v_d and sensed velocity noise, run the
    following plant and controller / compensation scenarios:

    - "free": Frictionless
    - "nc": Friction, No Compenstation
    - "ci": Friction, Compensation, Immediate
    - "cf": Friction, Compensation, Forceast

    """
    plant = MultibodyPlant(time_step=0.0)
    add_model(plant)
    plant.Finalize()

    joint = plant.GetJointByName("panda_joint7")
    Izz = get_rotational_axis_outboard_inertia(plant, joint)

    # Friction model.
    v0 = 0.05
    m = 0.8

    # Controller.
    kd = 0.5
    zoh_dt = 0.001

    # Compensation model's values.
    # N.B. Deviating from plant friction model means we may have steady
    # state error (if too lax) or too much noise (if too tight).
    comp_v0 = v0
    comp_m = m
    forecast_dt = zoh_dt

    def calc_friction(t, q, v):
        u = calc_friction_sample(t, q, v, v0=v0, m=m)
        return u

    calc_u = partial(calc_u_vel, kd=kd, v_d=v_d)

    def calc_u_comp_immediate(t, q, v):
        u = calc_u(t, q, v)
        u_friction = calc_friction_sample(t, q, v, v0=comp_v0, m=comp_m)
        return u - u_friction

    def calc_u_comp_forecast(t, q, v):
        u = calc_u(t, q, v)
        vd = u / Izz
        v_next = v + vd * forecast_dt
        u_friction = calc_friction_sample(t, q, v_next, v0=comp_v0, m=comp_m)
        return u - u_friction

    def run_scenario(calc_u_in, calc_friction_in):
        np.random.seed(0)
        log = run_friction_sim(
            calc_u_in,
            calc_friction_in,
            dt=zoh_dt,
            visualize=False,
            tf=0.2,
            noise_amp=np.array([0.0, v_noise]),
        )
        ts, _, vs, _ = extract_log_values(log)
        v_errs = vs - v_d
        return ts, v_errs

    # Frictionless.
    ts_free, ves_free = run_scenario(calc_u, no_friction)
    ts_nc, ves_nc = run_scenario(calc_u, calc_friction_sample)
    ts_ci, ves_ci = run_scenario(calc_u_comp_immediate, calc_friction_sample)
    ts_cf, ves_cf = run_scenario(calc_u_comp_forecast, calc_friction_sample)

    ts = ts_free
    assert_allclose(ts, ts_nc)
    assert_allclose(ts, ts_ci)
    assert_allclose(ts, ts_cf)

    return SimpleNamespace(
        ts=ts,
        ves_free=ves_free,
        ves_nc=ves_nc,
        ves_ci=ves_ci,
        ves_cf=ves_cf,
    )


def plot_compensation_experiments(title, info):
    _, axs = plt.subplots(
        nrows=4,
        ncols=1,
        sharex=True,
        sharey=True,
        num=title,
    )

    label_free = "Frictionless"
    label_nc = "Friction, No Compenstation"
    label_ci = "Friction, Compenstation, Immediate"
    label_cf = "Friction, Compenstation, Forecast"

    plt.sca(axs[0])
    plt.plot(info.ts, info.ves_free)
    plt.title(label_free)

    plt.sca(axs[1])
    plt.plot(info.ts, info.ves_nc)
    plt.title(label_nc)

    plt.sca(axs[2])
    plt.plot(info.ts, info.ves_ci)
    plt.ylabel("v err (actual - desired) [rad/s]")
    plt.title(label_ci)

    plt.sca(axs[3])
    plt.plot(info.ts, info.ves_cf)
    plt.xlabel("time [s]")
    plt.title(label_cf)

    plt.tight_layout()


def uniform_like(x, low, high):
    delta = torch.zeros_like(x)
    delta.uniform_(low, high)
    return delta


def make_fit_test_data(*, friction_gt, u_noise):
    # Generate data.
    vs = torch.linspace(-2, 2, 1000)
    us = friction_gt(vs)
    # Corrupt with noise.
    # TODO(eric.cousineau): Should add noise to velocities?
    us += uniform_like(us, -u_noise, u_noise)
    return vs, us


def assert_allclose(a, b, *, tol=0):
    np.testing.assert_allclose(a, b, rtol=0, atol=tol)


class Test(unittest.TestCase):
    def test_regularizers(self):
        """Checks basic sigmoid proprtiers."""
        ss = np.linspace(-3.0, 3.0, 1000)
        m = 0.8
        s_big = 1e5
        tol = 1e-5

        def check(name, r):
            assert_allclose(r(0), 0.0, tol=tol)
            assert_allclose(r(1), m, tol=tol)
            assert_allclose(r(-1), -m, tol=tol)
            assert_allclose(r(s_big), 1.0, tol=tol)
            assert_allclose(r(-s_big), -1.0, tol=tol)
            if VISUALIZE:
                plt.plot(ss, r(ss, m=m), label=name)

        check("arctan", partial(regularizer_arctan, m=m))
        check("tanh", partial(regularizer_tanh, m=m))

        if VISUALIZE:
            plt.legend()
            plt.show()

    def test_friction_experiments(self):
        """
        Ensures that we can recover (velocity, toruqe) curves for identifying
        Coulomb friction.
        """
        calc_friction = partial(calc_friction_sample, b=0.1)
        (vs_m, us_m) = run_friction_experiments(calc_friction)

        def friction_neg(v):
            # Negate friction to align with the control necessary to counteract
            # it.
            return -calc_friction(t=None, q=None, v=v)

        us_gt = friction_neg(vs_m)
        assert_allclose(us_m, us_gt, tol=5e-4)

        if VISUALIZE:
            # Recompute with smoother data.
            vs_gt = np.linspace(vs_m[0], vs_m[-1], 1000)
            us_gt = friction_neg(vs_gt)
            plt.plot(vs_gt, us_gt, label="gt")
            plt.plot(vs_m, us_m, "x", label="sim")
            plt.legend()
            plt.show()

    def test_compensation_experiments(self):
        """
        Tests controllers and friction compensation modes. See
        `run_compensation_example` for more details.
        """
        # TODO(eric.cousineau): Testing here's a bit brittle.

        v_noise = 0.01

        def run(title, v_d):
            info = run_compensation_example(v_d=v_d, v_noise=v_noise)
            if VISUALIZE:
                plot_compensation_experiments(title, info)
            return info

        if VISUALIZE:
            plt.show(block=False)

        def assert_stationary(ves, ve_min):
            self.assertLess(maxabs(ves), ve_min)

        zero = run(title="Stationary", v_d=0.0)
        ts = zero.ts

        # Crackly, but expected.
        assert_stationary(zero.ves_free, ve_min=v_noise)
        # Noise is effective removed by friction.
        assert_stationary(zero.ves_nc, ve_min=v_noise / 10)
        # Lots of extra energy when doing "immediate" compensation.
        assert_stationary(zero.ves_ci, ve_min=0.04)
        # Similar to frictionless case.
        assert_stationary(zero.ves_cf, ve_min=v_noise)

        def assert_steady_state(ves, t_rise, ve_min):
            rose_actual = (np.abs(ves) <= ve_min).reshape((-1,))
            t_rise_actual = ts[rose_actual][0]
            self.assertEqual(t_rise, t_rise_actual)
            self.assertLess(maxabs(ves[rose_actual]), ve_min)

        slow = run(title="Slow", v_d=0.1)
        assert_allclose(ts, slow.ts)

        # Crackly, but expected.
        assert_steady_state(slow.ves_free, ve_min=v_noise * 2, t_rise=0.002)
        # Yuck. Just crap (current state of affairs).
        assert_steady_state(slow.ves_nc, ve_min=0.0975, t_rise=0.001)
        # Slower b/c we're not forecasting.
        assert_steady_state(slow.ves_ci, ve_min=v_noise, t_rise=0.01)
        # Similar to frictionless case.
        assert_steady_state(slow.ves_cf, ve_min=v_noise, t_rise=0.002)

        fast = run(title="Fast", v_d=1.0)
        assert_allclose(ts, fast.ts)

        # Crackly, but expected.
        assert_steady_state(fast.ves_free, ve_min=v_noise * 2, t_rise=0.005)
        # Yuck. Still crap.
        assert_steady_state(fast.ves_nc, ve_min=0.8, t_rise=0.008)
        # Slower b/c we're not forecasting.
        assert_steady_state(fast.ves_ci, ve_min=0.04, t_rise=0.005)
        # Similar to frictionless case.
        assert_steady_state(fast.ves_cf, ve_min=v_noise * 2, t_rise=0.005)

        if VISUALIZE:
            plt.show(block=True)

    @torch.no_grad()
    def test_fit(self):
        """
        Tests fitting a friction model in an isolated context.
        See `run_fit_experiment` for more details.
        """
        # Ground-truth model.
        friction_gt = JointFriction(
            dry=JointDryFriction(v0=0.01, u_max=0.4), b=0.1
        )
        # Initial values.
        friction = JointFriction.make_initial_guess()
        # Error tolerances.
        errs_tol = {
            "dry.v0": 0.02,
            "dry.u_max": 0.005,
            "b": 0.005,
        }

        projected = True
        v0_min = 1e-8
        u_max_min = 0

        if not projected:
            friction.dry.remove_projection()

        @torch.no_grad()
        def ensure_valid_param(friction):
            dry = friction.dry
            v0 = param_value(dry.v0)
            u_max = param_value(dry.u_max)
            # m = param_value(dry.m)
            b = param_value(friction.b)
            if projected:
                assert v0 >= v0_min
                assert u_max >= u_max_min
                assert b >= 0
            else:
                # Yuck!
                dry.v0.data = torch.clip(v0, v0_min, np.inf)
                dry.u_max.data = torch.clip(u_max, u_max_min, np.inf)
                friction.b.data = torch.clip(b, 0, np.inf)

        # Run experiment.
        torch.manual_seed(0)
        vs, us = make_fit_test_data(friction_gt=friction_gt, u_noise=0.1)
        info = run_fit_experiment(
            vs=vs,
            us=us,
            friction=friction,
            ensure_valid_param=ensure_valid_param,
            lr=1e-1,
        )

        if VISUALIZE:
            plot_fit_experiment(info)
            plt.show()

        # Ensure loss is purty low.
        _, loss_per_epoch = np.asarray(info.loss_info).T
        self.assertGreater(loss_per_epoch[0], 0.05)
        self.assertLess(loss_per_epoch[-1], 0.004)

        # Ensure we got close enough to ground-truth values.
        param_gt = friction_gt.custom_named_parameters()
        param = friction.custom_named_parameters()
        items_iter = dict_items_zip(param_gt, param, errs_tol)
        for name, (gt, pred, tol) in items_iter:
            gt = param_value(gt)
            pred = param_value(pred)
            np.testing.assert_allclose(
                gt.numpy(),
                pred.numpy(),
                atol=tol,
                rtol=0.0,
                err_msg=name,
            )


if __name__ == "__main__":
    unittest.main()
