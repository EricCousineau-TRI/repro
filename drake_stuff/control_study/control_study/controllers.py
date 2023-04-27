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
from control_study.limits import PlantLimits
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
        tau = self.calc_control(pose_actual, pose_desired, q0)

        if self.check_limits:
            self.nominal_limits.assert_values_within_limits(u=tau, tol=tol)

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


def make_osqp_solver_and_options(use_dairlab_settings=True):
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
            # rho=0.001,
            # sigma=1e-6,
            # max_iter=1000,
            # max_iter=10000,
            # max_iter=500,
            # max_iter=250,
            # max_iter=10000,
            # eps_abs=1e-3,
            # eps_rel=1e-4,
            # eps_abs=5e-4,
            # eps_rel=5e-4,
            # eps_abs=1e-5,
            # eps_rel=1e-5,
            # eps_prim_inf=1e-5,
            # eps_dual_inf=1e-5,
            polish=1,
            polish_refine_iter=1,
            scaled_termination=1,
            scaling=1,
        )
    for name, value in solver_options_dict.items():
        solver_options.SetOption(solver_id, name, value)
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

    def calc_control(self, pose_actual, pose_desired, q0):
        M, C, tau_g = calc_dynamics(self.plant, self.context)
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
        add_simple_limits(
            self.plant_limits,
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

        Minv = inv(M)
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
        self.solver, self.solver_options = make_osqp_solver_and_options()
        self.acceleration_bounds_dt = acceleration_bounds_dt
        self.posture_weight = posture_weight
        self.use_torque_weights = use_torque_weights
        self.prev_sol = None

    def calc_control(self, pose_actual, pose_desired, q0):
        q = self.plant.GetPositions(self.context)
        v = self.plant.GetVelocities(self.context)
        num_v = len(v)
        M, C, tau_g = calc_dynamics(self.plant, self.context)
        Minv = inv(M)

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
        add_simple_limits(
            self.plant_limits,
            self.acceleration_bounds_dt,
            q,
            v,
            prog,
            vd_star,
            u_star,
        )

        # Compute spatial feedback.
        gains_t = self.gains.task
        num_t = 6
        It = np.eye(num_t)
        X, V, Jt, Jtdot_v = pose_actual
        X_des, V_des, A_des = pose_desired
        V_des = V_des.get_coeffs()
        A_des = A_des.get_coeffs()
        e = se3_vector_minus(X, X_des)
        ed = V - V_des
        edd_c = A_des - gains_t.kp * e - gains_t.kd * ed

        Mt, Mtinv, Jt, Jtbar, Nt_T = reproject_mass(Minv, Jt)

        kinematic = False
        if kinematic:
            Jtpinv = np.linalg.pinv(Jt)
            Nt_T = Iv - Jtpinv @ Jt

        # print(np.linalg.matrix_rank(Jt))
        # print(np.linalg.matrix_rank(Nt_T))

        # Constrain along desired tracking, J*vdot + Jdot*v = s*edd_c
        # For simplicity, allow each direction to have its own scaling.
        num_t = 6
        # scale_A = np.eye(num_t)
        # scale_A = np.ones((num_t, 1))
        scale_A = np.array([
            [1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1],
        ]).T
        num_scales = scale_A.shape[1]
        task_bias_rep = np.tile(edd_c, (num_scales, 1)).T
        scale_vars = prog.NewContinuousVariables(num_scales, "scale")
        task_vars = np.concatenate([vd_star, scale_vars])
        task_A = np.hstack([Jt, -scale_A * task_bias_rep])
        task_b = -Jtdot_v

        relax_primary = False
        relax_secondary = False
        # relax_penalty = 1e1
        # relax_penalty = 1e2
        # relax_penalty = 1e3
        # relax_penalty = 1e4
        # relax_penalty = 1e5
        # relax_penalty = 1e6
        if relax_primary:
            relax_vars = prog.NewContinuousVariables(num_t, "task.relax")
            task_vars = np.concatenate([task_vars, relax_vars])
            task_A = np.hstack([task_A, -It])
            if kinematic:
                proj = Jtpinv
            else:
                proj = Jt.T @ Mt
            prog.Add2NormSquaredCost(
                relax_penalty * proj @ It,
                proj @ np.zeros(num_t),
                relax_vars,
            )

        prog.AddLinearEqualityConstraint(
            task_A, task_b, task_vars
        ).evaluator().set_description("task")

        # Try to optimize towards scale=1.
        proj = np.eye(num_scales)
        # proj = Jt.T @ Mt @ scale_A
        # proj = Mt @ scale_A
        # proj = scale_A
        # import pdb; pdb.set_trace()
        # proj *= 100
        # proj = proj * np.sqrt(num_scales)
        desired_scales = np.ones(num_scales)
        prog.Add2NormSquaredCost(
            proj @ np.eye(num_scales),
            proj @ desired_scales,
            scale_vars,
        )

        edd_c_t = edd_c
        scale_t_vars = scale_vars

        # ones = np.ones(num_scales)
        # prog.AddBoundingBoxConstraint(
        #     -1.0 * ones,
        #     1.0 * ones,
        #     scale_vars,
        # )

        # Compute posture feedback.
        gains_p = self.gains.posture
        e = q - q0
        ed = v
        edd_c = -gains_p.kp * e - gains_p.kd * ed
        # Same as above, but lower weight.
        if self.use_torque_weights:
            task_proj = Nt_T
        else:
            task_proj = Iv

        # TODO(eric.cousineau): Maybe I need to constrain these error dynamics?

        # weight = self.posture_weight
        # task_proj = weight * task_proj
        # task_A = task_proj @ Iv
        # task_b = task_proj @ edd_c
        # prog.Add2NormSquaredCost(task_A, task_b, vd_star)

        # scale_A = np.ones((num_v, 1))
        scale_A = np.eye(num_v)
        num_scales = scale_A.shape[1]
        task_bias_rep = np.tile(edd_c, (num_scales, 1)).T
        scale_vars = prog.NewContinuousVariables(num_scales, "scale")
        task_vars = np.concatenate([vd_star, scale_vars])
        task_A = np.hstack([Iv, -scale_A * task_bias_rep])
        task_b = np.zeros(num_v)

        # TODO(eric.cousineau): Weigh penalty based on how much feedback we
        # need?

        if relax_secondary:
            relax_vars = prog.NewContinuousVariables(num_v, "q relax")
            task_vars = np.concatenate([task_vars, relax_vars])
            task_A = np.hstack([task_A, -Iv])
            proj = task_proj
            prog.Add2NormSquaredCost(
                relax_penalty * proj @ Iv,
                proj @ np.zeros(num_v),
                relax_vars,
            )

        prog.AddLinearEqualityConstraint(
            task_proj @ task_A,
            task_proj @ task_b,
            task_vars,
        ).evaluator().set_description("posture")
        desired_scales = np.ones(num_scales)
        proj = self.posture_weight * np.eye(num_scales)
        # proj = self.posture_weight * task_proj @ scale_A
        # proj = self.posture_weight * scale_A
        # proj = proj #/ np.sqrt(num_scales)
        prog.Add2NormSquaredCost(
            proj @ np.eye(num_scales),
            proj @ desired_scales,
            scale_vars,
        )

        # ones = np.ones(num_scales)
        # prog.AddBoundingBoxConstraint(
        #     1.0 * ones,
        #     1.0 * ones,
        #     scale_vars,
        # )

        edd_c_q = edd_c
        scale_q_vars = scale_vars

        # Solve.
        try:
            result = solve_or_die(
                self.solver, self.solver_options, prog, x0=self.prev_sol
            )
        except RuntimeError:
            print(np.rad2deg(self.plant_limits.q.lower))
            print(np.rad2deg(self.plant_limits.q.upper))
            print(np.rad2deg(q))
            print(self.plant_limits.v)
            print(v)
            raise

        infeas = result.GetInfeasibleConstraintNames(prog, tol=1e-2)
        assert len(infeas) == 0, "\n".join(infeas)
        self.prev_sol = result.get_x_val()

        tau = result.GetSolution(u_star)
        tau = self.plant_limits.u.saturate(tau)

        scale_t = result.GetSolution(scale_t_vars)
        scale_q = result.GetSolution(scale_q_vars)
        print(f"{scale_t}\n  {edd_c_t}")
        print(f"{scale_q}\n  {edd_c_q}")
        print("---")

        return tau
