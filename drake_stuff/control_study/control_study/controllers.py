import dataclasses as dc

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
    MathematicalProgram,
    OsqpSolver,
    SolverOptions,
)
from pydrake.systems.framework import LeafSystem

from control_study.geometry import se3_vector_minus
from control_study.spaces import declare_spatial_motion_inputs
from control_study.systems import declare_simple_init
from control_study.multibody_extras import calc_velocity_jacobian

from control_study.acceleration_bounds import compute_acceleration_bounds


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
        self.inputs_motion_desired = declare_spatial_motion_inputs(
            self,
            name_X="X_des",
            name_V="V_des",
            name_A="A_des",
        )
        self.torques_output = self.DeclareVectorOutputPort(
            "torques",
            size=self.num_q,
            calc=self.calc_torques,
        )
        self.get_init_state = declare_simple_init(
            self,
            self.on_init,
        )

    def on_init(self, sys_context, init):
        x = self.state_input.Eval(sys_context)
        self.plant.SetPositionsAndVelocities(self.context, x)
        q = self.plant.GetPositions(self.context)
        init.q = q

    def calc_torques(self, sys_context, output):
        x = self.state_input.Eval(sys_context)
        self.plant.SetPositionsAndVelocities(self.context, x)
        pose_actual = calc_spatial_values(
            self.plant, self.context, self.frame_W, self.frame_G
        )
        pose_desired = self.inputs_motion_desired.eval(sys_context)
        init = self.get_init_state(sys_context)
        q0 = init.q
        tau = self.calc_control(pose_actual, pose_desired, q0)
        output.set_value(tau)

    def calc_control(self, pose_actual, pose_desired, q0):
        raise NotImplementedError()


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

    def calc_control(self, pose_actual, pose_desired, q0):
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
            rho=0.001,
            sigma=1e-6,
            max_iter=250,
            eps_abs=1e-5,
            eps_rel=1e-5,
            eps_prim_inf=1e-5,
            eps_dual_inf=1e-5,
            polish=1,
            polish_refine_iter=1,
            scaled_termination=1,
            scaling=1,
        )
    for name, value in solver_options_dict.items():
        solver_options.SetOption(solver_id, name, value)
    return solver, solver_options


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
    ):
        super().__init__(plant, frame_W, frame_G)
        self.gains = gains
        self.plant_limits = plant_limits
        self.solver, self.solver_options = make_osqp_solver_and_options()
        self.acceleration_bounds_dt = acceleration_bounds_dt

    def calc_control(self, pose_actual, pose_desired, q0):
        M, C, tau_g = calc_dynamics(self.plant, self.context)

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

        # Compute spatial feedback.
        gains_t = self.gains.task
        X, V, Jt, Jtdot_v = pose_actual
        X_des, V_des, A_des = pose_desired
        V_des = V_des.get_coeffs()
        A_des = A_des.get_coeffs()
        e = se3_vector_minus(X, X_des)
        ed = V - V_des
        edd_c = A_des - gains_t.kp * e - gains_t.kd * ed
        # Drive towards desired tracking, |(J*vdot + Jdot*v) - (edd_c)|^2
        task_A = Jt
        task_b = -Jtdot_v + edd_c
        prog.Add2NormSquaredCost(task_A, task_b, vd_star)

        # Compute posture feedback.
        gains_p = self.gains.posture
        q = self.plant.GetPositions(self.context)
        v = self.plant.GetVelocities(self.context)
        e = q - q0
        ed = v
        edd_c = -gains_p.kp * e - gains_p.kd * ed
        # Same as above, but lower weight.
        weight = 0.01
        task_A = weight * Iv
        task_b = weight * edd_c
        prog.Add2NormSquaredCost(task_A, task_b, vd_star)

        # Add limits.
        vd_limits = compute_acceleration_bounds(
            q=q,
            v=v,
            plant_limits=self.plant_limits,
            dt=self.acceleration_bounds_dt,
        )
        if vd_limits.any_finite():
            vd_min, vd_max = vd_limits
            prog.AddBoundingBoxConstraint(
                vd_min, vd_max, vd_star
            ).evaluator().set_description("accel")

        # - Torque.
        u_limits = self.plant_limits.u
        if u_limits.any_finite():
            u_min, u_max = u_limits
            prog.AddBoundingBoxConstraint(
                u_min, u_max, u_star
            ).evaluator().set_description("torque")

        # Solve.
        result = self.solver.Solve(prog, solver_options=self.solver_options)
        if not result.is_success():
            self.solver_options.SetOption(
                CommonSolverOption.kPrintToConsole, True
            )
            self.solver.Solve(prog, solver_options=self.solver_options)
            print("\n".join(result.GetInfeasibleConstraintNames(prog)))
            print(result.get_solution_result())
            assert False, "Bad solution"
        tau = result.GetSolution(u_star)

        return tau


def solve_or_die(solver, solver_options, prog):
    result = solver.Solve(prog, solver_options=solver_options)
    if not result.is_success():
        solver_options.SetOption(
            CommonSolverOption.kPrintToConsole, True
        )
        solver.Solve(prog, solver_options=solver_options)
        print("\n".join(result.GetInfeasibleConstraintNames(prog)))
        print(result.get_solution_result())
        assert False, "Bad solution"
    return result


def add_simple_limits(plant_limits, dt, q, v, prog, vd_star, u_star):
    vd_limits = compute_acceleration_bounds(
        q=q,
        v=v,
        plant_limits=plant_limits,
        dt=dt,
    )
    if vd_limits.any_finite():
        vd_min, vd_max = vd_limits
        prog.AddBoundingBoxConstraint(
            vd_min, vd_max, vd_star
        ).evaluator().set_description("accel")

    # - Torque.
    if plant_limits.u.any_finite():
        u_min, u_max = plant_limits.u
        prog.AddBoundingBoxConstraint(
            u_min, u_max, u_star
        ).evaluator().set_description("torque")


class QpWithPrimaryConstraint(BaseController):
    def __init__(
        self,
        plant,
        frame_W,
        frame_G,
        *,
        gains,
        plant_limits,
        acceleration_bounds_dt,
        posture_weight=0.01,
    ):
        super().__init__(plant, frame_W, frame_G)
        self.gains = gains
        self.plant_limits = plant_limits
        self.solver, self.solver_options = make_osqp_solver_and_options()
        self.acceleration_bounds_dt = acceleration_bounds_dt
        self.posture_weight = posture_weight

    def calc_control(self, pose_actual, pose_desired, q0):
        q = self.plant.GetPositions(self.context)
        v = self.plant.GetVelocities(self.context)
        M, C, tau_g = calc_dynamics(self.plant, self.context)

        # Base QP formulation.
        Iv = np.eye(self.num_q)
        zv = np.zeros(self.num_q)
        prog = MathematicalProgram()
        vd_star = prog.NewContinuousVariables(self.num_q, "vd_star")
        u_star = prog.NewContinuousVariables(self.num_q, "u_star")

        # Add limits.
        add_simple_limits(
            self.plant_limits,
            self.acceleration_bounds_dt,
            q,
            v,
            prog,
            vd_star,
            u_star,
        )

        # Dynamics constraint.
        dyn_vars = np.concatenate([vd_star, u_star])
        dyn_A = np.hstack([M, -Iv])
        dyn_b = -C + tau_g
        prog.AddLinearEqualityConstraint(
            dyn_A, dyn_b, dyn_vars
        ).evaluator().set_description("dyn")

        # Compute spatial feedback.
        gains_t = self.gains.task
        X, V, Jt, Jtdot_v = pose_actual
        X_des, V_des, A_des = pose_desired
        V_des = V_des.get_coeffs()
        A_des = A_des.get_coeffs()
        e = se3_vector_minus(X, X_des)
        ed = V - V_des
        edd_c = A_des - gains_t.kp * e - gains_t.kd * ed
        # Drive towards desired tracking, |(J*vdot + Jdot*v) - (edd_c)|^2
        task_A = Jt
        task_b = -Jtdot_v + edd_c
        prog.Add2NormSquaredCost(task_A, task_b, vd_star)

        # Compute posture feedback.
        gains_p = self.gains.posture
        e = q - q0
        ed = v
        edd_c = -gains_p.kp * e - gains_p.kd * ed
        # Same as above, but lower weight.
        weight = self.posture_weight
        task_A = weight * Iv
        task_b = weight * edd_c
        prog.Add2NormSquaredCost(task_A, task_b, vd_star)

        # Solve.
        result = solve_or_die(self.solver, self.solver_options, prog)
        tau = result.GetSolution(u_star)

        return tau
