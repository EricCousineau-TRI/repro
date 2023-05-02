import dataclasses as dc
from textwrap import indent

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv

from pydrake.common.value import Value
from pydrake.math import RigidTransform
from pydrake.multibody.math import (
    SpatialAcceleration,
    SpatialForce,
    SpatialVelocity,
)
from pydrake.multibody.plant import ExternallyAppliedSpatialForce
from pydrake.multibody.tree import JacobianWrtVariable, ModelInstanceIndex
from pydrake.solvers import (
    ClpSolver,
    CommonSolverOption,
    GurobiSolver,
    MathematicalProgram,
    MosekSolver,
    OsqpSolver,
    ScsSolver,
    SnoptSolver,
    SolverOptions,
)
from pydrake.systems.framework import LeafSystem

from control_study.geometry import se3_vector_minus
from control_study.limits import PlantLimits, VectorLimits
from control_study.systems import declare_simple_init
from control_study.multibody_extras import calc_velocity_jacobian

from control_study.acceleration_bounds import compute_acceleration_bounds

SHOULD_STOP = False


class BaseController(LeafSystem):
    def __init__(self, plant, frame_W, frame_G):
        super().__init__()
        self.plant = plant
        self.frame_W = frame_W
        self.frame_G = frame_G
        self.context = plant.CreateDefaultContext()
        self.num_q = plant.num_positions()
        self.num_x = 2 * self.num_q
        assert plant.num_velocities() == self.num_q
        self.state_input = self.DeclareVectorInputPort("state", self.num_x)
        self.torques_output = self.DeclareVectorOutputPort(
            "torques",
            size=self.num_q,
            calc=self.calc_torques,
        )
        self.get_init_state = declare_simple_init(
            self,
            self.on_init,
        )
        self.check_limits = True
        self.nominal_limits = PlantLimits.from_plant(plant)
        # Will be set externally.
        self.traj = None

    def on_init(self, sys_context, init):
        x = self.state_input.Eval(sys_context)
        self.plant.SetPositionsAndVelocities(self.context, x)
        q = self.plant.GetPositions(self.context)
        init.q = q

    def calc_torques(self, sys_context, output):
        x = self.state_input.Eval(sys_context)
        t = sys_context.get_time()

        tol = 1e-4
        self.plant.SetPositionsAndVelocities(self.context, x)
        if self.check_limits:
            q = self.plant.GetPositions(self.context)
            v = self.plant.GetVelocities(self.context)
            self.nominal_limits.assert_values_within_limits(q=q, v=v, tol=tol)

        init = self.get_init_state(sys_context)
        q0 = init.q
        pose_actual = calc_spatial_values(
            self.plant, self.context, self.frame_W, self.frame_G
        )
        pose_desired = self.traj(t)
        tau = self.calc_control(t, pose_actual, pose_desired, q0)

        if self.check_limits:
            self.nominal_limits.assert_values_within_limits(u=tau, tol=tol)

        output.set_value(tau)

    def calc_control(self, t, pose_actual, pose_desired, q0):
        raise NotImplementedError()

    def show_plots(self):
        pass


@dc.dataclass
class Gains:
    kp: np.ndarray
    kd: np.ndarray

    @staticmethod
    def critically_damped(kp):
        kd = 2 * np.sqrt(kp)
        return Gains(kp, kd)


@dc.dataclass
class OscGains:
    task: Gains
    posture: Gains

    @staticmethod
    def critically_damped(kp_t, kp_p):
        return OscGains(
            Gains.critically_damped(kp_t),
            Gains.critically_damped(kp_p),
        )


def calc_spatial_values(plant, context, frame_W, frame_G):
    X = plant.CalcRelativeTransform(context, frame_W, frame_G)
    J, Jdot_v = calc_velocity_jacobian(
        plant, context, frame_W, frame_G, include_bias=True
    )
    v = plant.GetVelocities(context)
    V = J @ v
    return X, V, J, Jdot_v


def calc_dynamics(plant, context):
    M = plant.CalcMassMatrix(context)
    C = plant.CalcBiasTerm(context)
    tau_g = plant.CalcGravityGeneralizedForces(context)
    return M, C, tau_g


def reproject_mass(Minv, Jt):
    _, num_dof = Jt.shape
    I_dyn = np.eye(num_dof)
    # Maps from task forces to task accelerations.
    Mtinv = Jt @ Minv @ Jt.T
    # Maps from task accelerations to task forces.
    Mt = np.linalg.inv(Mtinv)
    # Maps from task accelerations to generalized accelerations.
    # Transpose maps from generalized forces to task forces.
    Jtbar = Minv @ Jt.T @ Mt
    # Generalized force nullspace.
    Nt_T = I_dyn - Jt.T @ Jtbar.T
    return Mt, Mtinv, Jt, Jtbar, Nt_T


class Osc(BaseController):
    """Explicit OSC."""
    def __init__(self, plant, frame_W, frame_G, gains):
        super().__init__(plant, frame_W, frame_G)
        self.gains = gains

    def calc_control(self, t, pose_actual, pose_desired, q0):
        M, C, tau_g = calc_dynamics(self.plant, self.context)
        Minv = inv(M)

        # Compute spatial feedback.
        gains_t = self.gains.task
        X, V, Jt, Jtdot_v = pose_actual
        X_des, V_des, A_des = pose_desired
        V_des = V_des.get_coeffs()
        A_des = A_des.get_coeffs()
        e = se3_vector_minus(X, X_des)
        ed = V - V_des
        edd_c = A_des - gains_t.kp * e - gains_t.kd * ed
        Mt, _, Jt, _, Nt_T = reproject_mass(Minv, Jt)
        Ft = Mt @ (edd_c - Jtdot_v)

        # Compute posture feedback.
        gains_p = self.gains.posture
        q = self.plant.GetPositions(self.context)
        v = self.plant.GetVelocities(self.context)
        e = q - q0
        ed = v
        edd_c = -gains_p.kp * e - gains_p.kd * ed
        Fp = M @ edd_c

        # Sum up tasks and cancel gravity + Coriolis terms.
        tau = Jt.T @ Ft + Nt_T @ Fp + C - tau_g
        return tau


def make_osqp_solver_and_options(use_dairlab_settings=False):
    solver = OsqpSolver()
    solver_id = solver.solver_id()
    solver_options = SolverOptions()
    # https://osqp.org/docs/interfaces/solver_settings.html#solver-settings
    solver_options_dict = dict(
        # See https://github.com/RobotLocomotion/drake/issues/18711
        adaptive_rho=0,
    )
    if use_dairlab_settings:
        # https://github.com/DAIRLab/dairlib/blob/0da42bc2/examples/Cassie/osc_run/osc_running_qp_settings.yaml
        solver_options_dict.update(
            rho=1.0,
            sigma=1.0,
            # alpha=1.9,
            # sigma=100,  # er, int values messes things up?
            # rho=0.1,
            # sigma=1e-6,
            # max_iter=250,
            # max_iter=500,
            max_iter=1000,
            # max_iter=2000,
            # max_iter=10000,
            # eps_abs=1e-3,
            # eps_rel=1e-4,
            # eps_abs=5e-4,
            # eps_rel=5e-4,
            # eps_abs=1e-5,
            # eps_rel=1e-5,
            # eps_abs=1e-6,
            # eps_rel=1e-6,
            # eps_prim_inf=1e-5,
            # eps_dual_inf=1e-5,
            polish=1,
            polish_refine_iter=2,
            scaled_termination=1,
            scaling=1,
        )
    for name, value in solver_options_dict.items():
        solver_options.SetOption(solver_id, name, value)
    return solver, solver_options


def make_clp_solver_and_options():
    solver = ClpSolver()
    solver_options = SolverOptions()
    return solver, solver_options


def make_gurobi_solver_and_options():
    solver = GurobiSolver()
    solver_options = SolverOptions()
    return solver, solver_options


def make_mosek_solver_and_options():
    solver = MosekSolver()
    solver_options = SolverOptions()
    return solver, solver_options


def make_snopt_solver_and_options():
    solver = SnoptSolver()
    solver_options = SolverOptions()
    return solver, solver_options

def make_scs_solver_and_options():
    solver = ScsSolver()
    solver_options = SolverOptions()
    return solver, solver_options


def solve_or_die(solver, solver_options, prog, *, x0=None):
    result = solver.Solve(
        prog, solver_options=solver_options, initial_guess=x0
    )
    if not result.is_success():
        solver_options.SetOption(
            CommonSolverOption.kPrintToConsole, True
        )
        solver.Solve(prog, solver_options=solver_options)
        print("\n".join(result.GetInfeasibleConstraintNames(prog)))
        print(result.get_solution_result())
        raise RuntimeError("Bad solution")
    return result


def vd_limits_from_tau(u_limits, Minv, H):
    num_u = len(H)
    if not u_limits.isfinite():
        return VectorLimits(
            lower=-np.inf * np.ones(num_u),
            upper=np.inf * np.ones(num_u),
        )
    u_min, u_max = u_limits
    vd_tau_limits = VectorLimits(
        lower=Minv @ (u_min - H),
        upper=Minv @ (u_max - H),
    )
    # TODO(eric.cousineau): Is this even right? How to handle sign flip?
    vd_tau_limits = vd_tau_limits.make_valid()
    # assert vd_tau_limits.is_valid()
    return vd_tau_limits


def add_simple_limits(
    *,
    plant_limits,
    vd_limits,
    dt,
    q,
    v,
    prog,
    vd_vars,
    Avd,
    bvd,
    u_vars,
    Au,
    bu,
):
    # mode = "naive"
    mode = "cbf"
    # mode = "intersect"

    if mode == "naive":
        # v_next = v + dt*vd = Av*vd + bv
        # Av = dt * Avd
        # bv = v
        # v_rescale = 1 / dt
        Av = Avd
        v_to_vd = 1 / dt
        bv = v_to_vd * v
        # q_next = q + dt*v + 1/2*dt^2*vd = Aq*vd + bq
        # Aq = 0.5 * dt * dt * Avd
        # bq = q + dt * v
        # q_rescale = 1 / dt  # 2 / (dt * dt)
        Aq = Avd
        q_to_vd = 2 / (dt * dt)
        bq = q_to_vd * q + 2 / dt * v

        if plant_limits.q.any_finite():
            q_min, q_max = plant_limits.q
            prog.AddLinearConstraint(
                Avd,
                q_to_vd * q_min - bq,
                q_to_vd * q_max - bq,
                vd_vars,
            ).evaluator().set_description("pos")

        if plant_limits.v.any_finite():
            v_min, v_max = plant_limits.v
            prog.AddLinearConstraint(
                Avd,
                v_to_vd * v_min - bv,
                v_to_vd * v_max - bv,
                vd_vars,
            ).evaluator().set_description("vel")
    elif mode == "cbf":
        q_min, q_max = plant_limits.q
        v_min, v_max = plant_limits.v

        num_v = len(v)
        Iv = np.eye(num_v)

        # CBF formulation.
        # Goal: h >= 0 for all admissible states
        # hdd = c*vd >= -k_1*h -k_2*hd = b

        # Gains corresponding to naive formulation.
        aq_1 = lambda x: x
        aq_2 = aq_1
        av_1 = lambda x: x
        kq_1 = 2 / (dt * dt)
        kq_2 = 2 / dt
        kv_1 = 1 / dt

        dt_scale = 10
        kq_1 /= dt_scale**2
        kq_2 /= dt_scale
        kv_1 /= dt_scale

        # # Hacks
        # # kq_1 /= 100
        # # kq_2 /= 10
        # # kv_1 /= 10
        # kq_1 = 100
        # # kq_1 = 75
        # # kq_1 = 50
        # kq_2 = 50
        # kv_1 = 10
        # # kv_1 = 2

        # q_min
        h_q_min = q - q_min
        hd_q_min = v
        c_q_min = 1
        b_q_min = -kq_1 * aq_1(h_q_min) - kq_2 * aq_2(hd_q_min)
        # q_max
        h_q_max = q_max - q
        hd_q_max = -v
        c_q_max = -1
        b_q_max = -kq_1 * aq_1(h_q_max) - kq_2 * aq_2(hd_q_max)
        # v_min
        h_v_min = v - v_min
        c_v_min = 1
        b_v_min = -kv_1 * av_1(h_v_min)
        # v_max
        h_v_max = v_max - v
        c_v_max = -1
        b_v_max = -kv_1 * av_1(h_v_max)

        # Add constraints.
        # N.B. Nominal CBFs (c*vd >= b) are lower bounds. For CBFs where c=-1,
        # we can pose those as upper bounds (vd <= -b).
        prog.AddLinearConstraint(
            Avd,
            b_q_min,
            -b_q_max,
            vd_vars,
        ).evaluator().set_description("pos cbf")
        prog.AddLinearConstraint(
            Avd,
            b_v_min,
            -b_v_max,
            vd_vars,
        ).evaluator().set_description("vel cbf")
    elif mode == "intersect":
        vd_limits = compute_acceleration_bounds(
            q=q,
            v=v,
            plant_limits=plant_limits,
            dt=dt,
            vd_limits_nominal=vd_limits,
            check=False,
        )
    else:
        assert False

    # HACK - how to fix this?
    # vd_limits = vd_limits.make_valid()

    if vd_limits.any_finite():
        vd_min, vd_max = vd_limits
        prog.AddLinearConstraint(
            Avd,
            vd_min - bvd,
            vd_max - bvd,
            vd_vars,
        ).evaluator().set_description("accel")

    # - Torque.
    if u_vars is not None and plant_limits.u.any_finite():
        u_min, u_max = plant_limits.u
        prog.AddLinearConstraint(
            Au,
            u_min - bu,
            u_max - bu,
            u_vars,
        ).evaluator().set_description("torque")


class QpWithCosts(BaseController):
    def __init__(
        self,
        plant,
        frame_W,
        frame_G,
        *,
        gains,
        plant_limits,
        acceleration_bounds_dt,
        posture_weight,
        split_costs=None,
        use_torque_weights=False,
    ):
        super().__init__(plant, frame_W, frame_G)
        self.gains = gains
        self.plant_limits = plant_limits
        self.solver, self.solver_options = make_osqp_solver_and_options()
        self.acceleration_bounds_dt = acceleration_bounds_dt
        self.posture_weight = posture_weight
        self.split_costs = split_costs
        self.use_torque_weights = use_torque_weights

    def calc_control(self, t, pose_actual, pose_desired, q0):
        M, C, tau_g = calc_dynamics(self.plant, self.context)
        Minv = inv(M)

        q = self.plant.GetPositions(self.context)
        v = self.plant.GetVelocities(self.context)

        # Base QP formulation.
        Iv = np.eye(self.num_q)
        zv = np.zeros(self.num_q)
        prog = MathematicalProgram()

        vd_star = prog.NewContinuousVariables(self.num_q, "vd_star")
        u_star = prog.NewContinuousVariables(self.num_q, "u_star")

        # Dynamics constraint.
        dyn_vars = np.concatenate([vd_star, u_star])
        dyn_A = np.hstack([M, -Iv])
        dyn_b = -C + tau_g
        prog.AddLinearEqualityConstraint(
            dyn_A, dyn_b, dyn_vars
        ).evaluator().set_description("dyn")

        # Add limits.
        vd_limits = self.plant_limits.vd
        # vd_limits = intersect_vd_limits(
        #     self.plant_limits,
        #     Minv,
        #     C,
        #     tau_g,
        # )
        add_simple_limits(
            self.plant_limits,
            vd_limits,
            self.acceleration_bounds_dt,
            q,
            v,
            prog,
            vd_star,
            u_star,
        )

        # Compute spatial feedback.
        gains_t = self.gains.task
        X, V, Jt, Jtdot_v = pose_actual
        X_des, V_des, A_des = pose_desired
        V_des = V_des.get_coeffs()
        A_des = A_des.get_coeffs()
        e = se3_vector_minus(X, X_des)
        ed = V - V_des
        edd_c = A_des - gains_t.kp * e - gains_t.kd * ed

        Mt, Mtinv, Jt, Jtbar, Nt_T = reproject_mass(Minv, Jt)

        # Drive towards desired tracking, |(J*vdot + Jdot*v) - (edd_c)|^2
        task_A = Jt
        task_b = -Jtdot_v + edd_c

        num_t = 6
        It = np.eye(num_t)
        if self.use_torque_weights:
            # task_proj = Jt.T @ Mt
            task_proj = Mt
        else:
            task_proj = It
        task_A = task_proj @ task_A
        task_b = task_proj @ task_b
        if self.split_costs is None:
            prog.Add2NormSquaredCost(task_A, task_b, vd_star)
        else:
            slices = [slice(0, 3), slice(3, 6)]
            for weight_i, slice_i in zip(self.split_costs, slices):
                prog.Add2NormSquaredCost(
                    weight_i * task_A[slice_i],
                    weight_i* task_b[slice_i],
                    vd_star,
                )

        # Compute posture feedback.
        gains_p = self.gains.posture
        e = q - q0
        ed = v
        edd_c = -gains_p.kp * e - gains_p.kd * ed
        # Same as above, but lower weight.
        weight = self.posture_weight
        if self.use_torque_weights:
            task_proj = weight * Nt_T
        else:
            task_proj = weight * Iv
        task_A = task_proj
        task_b = task_proj @ edd_c
        prog.Add2NormSquaredCost(task_A, task_b, vd_star)

        # Solve.
        result = solve_or_die(self.solver, self.solver_options, prog)
        tau = result.GetSolution(u_star)

        return tau


class QpWithDirConstraint(BaseController):
    def __init__(
        self,
        plant,
        frame_W,
        frame_G,
        *,
        gains,
        plant_limits,
        acceleration_bounds_dt,
        posture_weight,
        use_torque_weights=False,
    ):
        super().__init__(plant, frame_W, frame_G)
        self.gains = gains
        self.plant_limits = plant_limits

        # Can be a bit imprecise, but w/ tuning can improve.
        # self.solver, self.solver_options = make_osqp_solver_and_options()

        # Best, it seems like?
        self.solver, self.solver_options = make_snopt_solver_and_options()

        # self.solver, self.solver_options = make_clp_solver_and_options()

        # self.solver, self.solver_options = make_gurobi_solver_and_options()

        # self.solver, self.solver_options = make_scs_solver_and_options()

        # Infeasible for implicit=True. Good for implicit=False.
        # self.solver, self.solver_options = make_mosek_solver_and_options()

        self.acceleration_bounds_dt = acceleration_bounds_dt
        self.posture_weight = posture_weight
        self.use_torque_weights = use_torque_weights
        self.prev_sol = None

        self.ts = []
        self.qs = []
        self.vs = []
        self.us = []
        self.edd_ts = []
        self.s_ts = []
        self.r_ts = []
        self.edd_ps = []
        self.edd_ps_null = []
        self.s_ps = []
        # self.r_ps = []
        self.limits_infos = []

    def calc_control(self, t, pose_actual, pose_desired, q0):
        q = self.plant.GetPositions(self.context)
        v = self.plant.GetVelocities(self.context)
        num_v = len(v)
        M, C, tau_g = calc_dynamics(self.plant, self.context)
        Minv = inv(M)
        H = C - tau_g

        # Base QP formulation.
        Iv = np.eye(self.num_q)
        zv = np.zeros(self.num_q)
        prog = MathematicalProgram()

        X, V, Jt, Jtdot_v = pose_actual
        Mt, Mtinv, Jt, Jtbar, Nt_T = reproject_mass(Minv, Jt)

        # Compute spatial feedback.
        gains_t = self.gains.task
        num_t = 6
        It = np.eye(num_t)
        X_des, V_des, A_des = pose_desired
        V_des = V_des.get_coeffs()
        A_des = A_des.get_coeffs()
        e = se3_vector_minus(X, X_des)
        ed = V - V_des
        edd_c_t = A_des - gains_t.kp * e - gains_t.kd * ed

        # Compute posture feedback.
        gains_p = self.gains.posture
        e = q - q0
        ed = v
        edd_c_p = -gains_p.kp * e - gains_p.kd * ed

        num_t = 6
        # scale_A_t = np.eye(num_t)
        # scale_A_t = np.ones((num_t, 1))
        scale_A_t = np.array([
            [1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1],
        ]).T
        num_scales_t = scale_A_t.shape[1]
        scale_vars_t = prog.NewContinuousVariables(num_scales_t, "scale_t")

        # If True, will add (vd, tau) as decision variables, and impose
        # dynamics and task acceleration constraints. If False, will explicitly
        # project to torques / accelerations.
        implicit = True

        # If True, will also scale secondary objective (posture).
        scale_secondary = True

        # For implicit formulation.
        dup_eq_as_cost = False
        dup_scale = 0.1

        relax_primary = None
        # relax_primary = 1.0
        # relax_primary = 5.0
        # relax_primary = 1e1
        # relax_primary = 1e1
        # relax_primary = np.array([100, 100, 100, 50, 50, 50])
        # relax_primary = np.array([20, 20, 20, 10, 10, 10])
        # relax_primary = 1e2
        # relax_primary = 1e3
        # relax_primary = 1e4
        # relax_primary = 1e5
        # relax_primary = 1e6
        relax_secondary = None
        # relax_secondary = relax_primary

        kinematic = False

        if scale_secondary:
            scale_A_p = np.ones((num_v, 1))
            # scale_A_p = np.eye(num_v)
            num_scales_p = scale_A_p.shape[1]
            scale_vars_p = prog.NewContinuousVariables(num_scales_p, "scale_p")

        assert self.use_torque_weights
        proj_t = Jt.T @ Mt
        proj_p = Nt_T @ M
        # proj_p = Nt_T

        if implicit:
            vd_star = prog.NewContinuousVariables(self.num_q, "vd_star")
            u_star = prog.NewContinuousVariables(self.num_q, "u_star")

            # Dynamics constraint.
            dyn_vars = np.concatenate([vd_star, u_star])
            dyn_A = np.hstack([M, -Iv])
            dyn_b = -H
            prog.AddLinearEqualityConstraint(
                dyn_A, dyn_b, dyn_vars
            ).evaluator().set_description("dyn")

            u_vars = u_star
            Au = np.eye(num_v)
            bu = np.zeros(num_v)
            vd_vars = vd_star
            Avd = np.eye(num_v)
            bvd = np.zeros(num_v)
        else:
            assert relax_primary is None
            assert relax_secondary is None
            Au_t = proj_t @ np.diag(edd_c_t) @ scale_A_t
            bu = -proj_t @ Jtdot_v + H
            if scale_secondary:
                Au_p = proj_p @ np.diag(edd_c_p) @ scale_A_p
                u_vars = np.concatenate([scale_vars_t, scale_vars_p])
                Au = np.hstack([Au_t, Au_p])
            else:
                Au = Au_t
                bu += proj_p @ edd_c_p
                u_vars = scale_vars_t
            vd_vars = u_vars
            Avd = Minv @ Au
            bvd = Minv @ (bu - H)

        # Add limits.
        vd_limits = self.plant_limits.vd
        # TODO(eric.cousineau): How to make this work correctly? Even
        # conservative estimate?
        torque_direct = True
        if torque_direct:
            u_vars = u_vars
        else:
            u_vars = None
            vd_tau_limits = vd_limits_from_tau(self.plant_limits.u, Minv, H)
            vd_limits = vd_limits.intersection(vd_tau_limits)
        limit_info = add_simple_limits(
            plant_limits=self.plant_limits,
            vd_limits=vd_limits,
            dt=self.acceleration_bounds_dt,
            q=q,
            v=v,
            prog=prog,
            vd_vars=vd_vars,
            Avd=Avd,
            bvd=bvd,
            u_vars=u_vars,
            Au=Au,
            bu=bu,
        )
        # if implicit:
        #     # prog.AddBoundingBoxConstraint(
        #     #     self.plant_limits.u.lower,
        #     #     self.plant_limits.u.upper,
        #     #     u_star,
        #     # ).evaluator().set_description("u direct")
        #     prog.AddBoundingBoxConstraint(
        #         vd_tau_limits.lower,
        #         vd_tau_limits.upper,
        #         vd_star,
        #     ).evaluator().set_description("u via vd")
        # else:
        #     # u_min, u_max = self.plant_limits.u
        #     # prog.AddLinearConstraint(
        #     #     Au,
        #     #     u_min - bu,
        #     #     u_max - bu,
        #     #     vd_vars,
        #     # ).evaluator().set_description("u direct")
        #     vd_min, vd_max = vd_tau_limits
        #     prog.AddLinearConstraint(
        #         Avd,
        #         vd_min - bvd,
        #         vd_max - bvd,
        #         vd_vars,
        #     ).evaluator().set_description("u via vd")

        if kinematic:
            Jtpinv = np.linalg.pinv(Jt)
            Nt_T = Iv - Jtpinv @ Jt

        # print(np.linalg.matrix_rank(Jt))
        # print(np.linalg.matrix_rank(Nt_T))

        # Constrain along desired tracking, J*vdot + Jdot*v = s*edd_c
        # For simplicity, allow each direction to have its own scaling.

        if implicit:
            task_vars_t = np.concatenate([vd_star, scale_vars_t])
            task_bias_t = edd_c_t
            task_A_t = np.hstack([Jt, -np.diag(task_bias_t) @ scale_A_t])
            task_b_t = -Jtdot_v

            if relax_primary is not None:
                relax_vars_t = prog.NewContinuousVariables(num_t, "task.relax")
                task_vars_t = np.concatenate([task_vars_t, relax_vars_t])
                task_A_t = np.hstack([task_A_t, -It])
                # proj = proj_t
                proj = np.diag(np.ones(num_t) * relax_primary)
                if kinematic:
                    proj = Jtpinv @ proj
                prog.Add2NormSquaredCost(
                    proj @ It,
                    proj @ np.zeros(num_t),
                    relax_vars_t,
                )

            prog.AddLinearEqualityConstraint(
                task_A_t, task_b_t, task_vars_t
            ).evaluator().set_description("task")

            if dup_eq_as_cost:
                prog.Add2NormSquaredCost(
                    dup_scale * task_A_t, dup_scale * task_b_t, task_vars_t
                )

        # Try to optimize towards scale=1.
        proj = np.eye(num_scales_t)
        # proj = Jt.T @ Mt @ scale_A
        # proj = Mt @ scale_A
        # proj = scale_A
        # import pdb; pdb.set_trace()
        # proj *= 100
        # proj = proj * np.sqrt(num_scales)
        desired_scales = np.ones(num_scales_t)
        prog.Add2NormSquaredCost(
            proj @ np.eye(num_scales_t),
            proj @ desired_scales,
            scale_vars_t,
        )

        # ones = np.ones(num_scales)
        # prog.AddBoundingBoxConstraint(
        #     -1.0 * ones,
        #     1.0 * ones,
        #     scale_vars,
        # )

        # TODO(eric.cousineau): Maybe I need to constrain these error dynamics?

        # weight = self.posture_weight
        # task_proj = weight * task_proj
        # task_A = task_proj @ Iv
        # task_b = task_proj @ edd_c
        # prog.Add2NormSquaredCost(task_A, task_b, vd_star)

        if implicit:
            if not scale_secondary:
                assert not dup_eq_as_cost
                task_A_p = proj_p
                task_b_p = proj_p @ edd_c_p
                prog.AddLinearEqualityConstraint(
                    task_A_p, task_b_p, vd_star,
                ).evaluator().set_description("posture")
            else:
                task_bias_p = edd_c_p
                # task_bias_rep = np.tile(edd_c, (num_scales, 1)).T
                task_vars_p = np.concatenate([vd_star, scale_vars_p])
                task_A_p = np.hstack([Iv, -np.diag(task_bias_p) @ scale_A_p])
                task_b_p = np.zeros(num_v)
                # TODO(eric.cousineau): Weigh penalty based on how much feedback we
                # need?
                if relax_secondary is not None:
                    relax_vars_p = prog.NewContinuousVariables(num_v, "q relax")
                    task_vars_p = np.concatenate([task_vars_p, relax_vars_p])
                    task_A_p = np.hstack([task_A_p, -Iv])
                    proj = proj_p
                    prog.Add2NormSquaredCost(
                        relax_secondary * proj @ Iv,
                        proj @ np.zeros(num_v),
                        relax_secondary,
                    )
                task_A_p = proj_p @ task_A_p
                task_b_p = proj_p @ task_b_p
                prog.AddLinearEqualityConstraint(
                    task_A_p, task_b_p, task_vars_p,
                ).evaluator().set_description("posture")
                if dup_eq_as_cost:
                    prog.Add2NormSquaredCost(
                        dup_scale * task_A_p, dup_scale * task_b_p, task_vars_p,
                    )

        if scale_secondary:
            desired_scales_p = np.ones(num_scales_p)
            proj = self.posture_weight * np.eye(num_scales_p)
            # proj = self.posture_weight * task_proj @ scale_A
            # proj = self.posture_weight * scale_A
            # proj = proj #/ np.sqrt(num_scales)
            prog.Add2NormSquaredCost(
                proj @ np.eye(num_scales_p),
                proj @ desired_scales_p,
                scale_vars_p,
            )

        # ones = np.ones(num_scales)
        # prog.AddBoundingBoxConstraint(
        #     1.0 * ones,
        #     1.0 * ones,
        #     scale_vars,
        # )

        # Solve.
        try:
            # TODO(eric.cousineau): OSQP does not currently accept
            # warm-starting:
            # https://github.com/RobotLocomotion/drake/blob/v1.15.0/solvers/osqp_solver.cc#L335-L336
            result = solve_or_die(
                self.solver, self.solver_options, prog, x0=self.prev_sol
            )
        except RuntimeError:
            # print(np.rad2deg(self.plant_limits.q.lower))
            # print(np.rad2deg(self.plant_limits.q.upper))
            # print(np.rad2deg(q))
            # print(self.plant_limits.v)
            # print(v)
            raise

        if implicit:
            # tol = 1e-3
            # tol = 1e-12  # snopt default
            # tol = 1e-10  # snopt, singular
            tol = 1e-6
            # tol = 1e-11  # osqp default
            # tol = 1e-3  # scs default
        else:
            # tol = 1e-14  # osqp, clp default
            # tol = 0  # gurobi default
            # tol = 1e-14  # snopt default
            tol = 1e-10  # snopt, singular
            # tol = 1e-14  # scs - primal infeasible, dual unbounded?
            # tol = 1e-11  # mosek default
            # tol = 1e-1  # HACK

        scale_t = result.GetSolution(scale_vars_t)

        # print(v)
        print(f"t: {scale_t}\n  {edd_c_t}")
        if relax_primary is not None:
            relax_t = result.GetSolution(relax_vars_t)
            print(f"  relax: {relax_t}")
        if scale_secondary:
            scale_p = result.GetSolution(scale_vars_p)
            print(f"p: {scale_p}\n  {edd_c_p}")
        if relax_secondary is not None:
            relax_p = result.GetSolution(relax_vars_p)
            print(f"  relax: {relax_p}")
        print("---")

        infeas = result.GetInfeasibleConstraintNames(prog, tol=tol)
        infeas_text = "\n" + indent("\n".join(infeas), "  ")
        assert len(infeas) == 0, infeas_text
        self.prev_sol = result.get_x_val()

        if implicit:
            tau = result.GetSolution(u_star)
        else:
            if scale_secondary:
                scale = np.concatenate([scale_t, scale_p])
            else:
                scale = scale_t
            tau = Au @ scale + bu
        tau = self.plant_limits.u.saturate(tau)

        # import pdb; pdb.set_trace()

        edd_c_p_null = Minv @ Nt_T @ M @ edd_c_p

        self.ts.append(t)
        self.qs.append(q)
        self.vs.append(v)
        self.us.append(tau)
        self.edd_ts.append(edd_c_t)
        self.s_ts.append(scale_t)
        if relax_primary is not None:
            self.r_ts.append(relax_t)
        self.edd_ps.append(edd_c_p)
        self.edd_ps_null.append(edd_c_p_null)
        self.s_ps.append(scale_p)
        self.limits_infos.append(limit_info)

        return tau

    def show_plots(self):
        ts = np.array(self.ts)
        qs = np.array(self.qs)
        vs = np.array(self.vs)
        us = np.array(self.us)
        edd_ts = np.array(self.edd_ts)
        s_ts = np.array(self.s_ts)
        r_ts = np.array(self.r_ts)
        edd_ps = np.array(self.edd_ps)
        edd_ps_null = np.array(self.edd_ps_null)
        s_ps = np.array(self.s_ps)

        # sub = slice(3, 4)
        sub = slice(None, None)
        sel = (slice(None, None), sub)

        def plot_lim(limits):
            lower, upper = limits
            lower = lower[sub]
            upper = upper[sub]
            ts_lim = ts[[0, -1]]
            reset_color_cycle()
            plt.plot(ts_lim, [lower, lower], ":")
            reset_color_cycle()
            plt.plot(ts_lim, [upper, upper], ":")

        _, axs = plt.subplots(num=1, nrows=3)
        plt.sca(axs[0])
        plt.plot(ts, qs[sel])
        legend_for(qs[sel])
        plot_lim(self.plant_limits.q)
        plt.title("q")
        plt.sca(axs[1])
        plt.plot(ts, vs[sel])
        legend_for(vs[sel])
        plot_lim(self.plant_limits.v)
        plt.title("v")
        plt.sca(axs[2])
        plt.plot(ts, us[sel])
        legend_for(us[sel])
        plot_lim(self.plant_limits.u)
        plt.title("u")

        plt.show()
        return  # HACK

        _, axs = plt.subplots(num=2, nrows=3)
        plt.sca(axs[0])
        plt.plot(ts, edd_ts)
        legend_for(edd_ts)
        plt.title("edd_c_t")
        plt.sca(axs[1])
        plt.plot(ts, edd_ps)
        legend_for(edd_ps)
        plt.title("edd_c_p")
        plt.sca(axs[2])
        plt.plot(ts, edd_ps_null)
        legend_for(edd_ps_null)
        plt.title("edd_c_p null")

        _, axs = plt.subplots(num=3, nrows=3)
        plt.sca(axs[0])
        plt.plot(ts, s_ts)
        legend_for(s_ts)
        plt.title("s_t")
        plt.sca(axs[1])
        plt.plot(ts, r_ts)
        legend_for(r_ts)
        plt.title("r_t")
        plt.sca(axs[2])
        plt.plot(ts, s_ps)
        legend_for(s_ps)
        plt.title("s_p")

        plt.show()


def legend_for(xs):
    n = xs.shape[1]
    labels = [f"{i}" for i in range(1, n + 1)]
    plt.legend(labels)


def reset_color_cycle():
    # https://stackoverflow.com/a/39116381/7829525
    plt.gca().set_prop_cycle(None)
