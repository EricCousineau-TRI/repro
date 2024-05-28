import numpy as np

from pydrake.all import (
    CommonSolverOption,
    JacobianWrtVariable,
    MathematicalProgram,
    OsqpSolver,
    SolverOptions,
)

from diff_ik_collision_avoidance.pose_util import se3_vector_minus


def SolveOrDie(solver, solver_options, prog, *, tol, x0=None):
    """
    Solves a program; if it does not report success, or if solution
    constraints violate beyond a specified tolerance, it will re-solve the
    problem with some additional debug information enabled.
    """
    result = solver.Solve(
        prog, solver_options=solver_options, initial_guess=x0
    )
    infeasible = result.GetInfeasibleConstraintNames(prog, tol=tol)
    if not result.is_success() or len(infeasible) > 0:
        # TODO(eric.cousineau): Print out max violation.
        print(f"Infeasible constraints per Drake for tol={tol}:")
        print("\n".join(infeasible))
        print()
        print("Re-solving with verbose output")
        solver_options.SetOption(CommonSolverOption.kPrintToConsole, True)
        result = solver.Solve(prog, solver_options=solver_options)
        print(result.get_solution_result())
        raise RuntimeError("Solver reports failure")
    return result


def CalcNullspace(J):
    n = J.shape[1]
    eye = np.eye(n)
    Jpinv = np.linalg.pinv(J)
    N = eye - Jpinv @ J
    return N


class JointLimits:
    def __init__(self, plant):
        self.position_lower = plant.GetPositionLowerLimits()
        self.position_upper = plant.GetPositionUpperLimits()
        self.velocity_lower = plant.GetVelocityLowerLimits()
        self.velocity_upper = plant.GetVelocityUpperLimits()


# TODO(eric.cousineau): Replace with results from
# https://github.com/RobotLocomotion/drake/pull/21487
class DiffIk:
    """
    Loosely represents a L2-norm-squared cost version of Drake's differential
    inverse kinematics:
    https://drake.mit.edu/doxygen_cxx/group__planning__kinematics.html
    Note that the current (v1.29.0) version will constrain along the direction
    of desired velocity, while this will allow us to get "close to" the
    velocity.

    This formulation is not yet perfect; consider tracking:
    https://github.com/RobotLocomotion/drake/pull/21487

    WARNING: This version is not very Systems-based.
    """

    def __init__(
        self,
        collision_checker,
        frame_F,
        dt,
        q0,
        full_formulation=False
    ):
        self.plant = collision_checker.plant()
        self.collision_checker = collision_checker
        self.dt = dt
        self.q0 = q0
        self.full_formulation = full_formulation

        self.frame_W = self.plant.world_frame()
        self.frame_F = frame_F
        self.joint_limits = JointLimits(self.plant)
        self.context = self.plant.CreateDefaultContext()

        self.solver = OsqpSolver()
        self.solver_options = SolverOptions()

    def solve(self, q, X_WF_des):
        prog = MathematicalProgram()
        num_v = self.plant.num_velocities()
        v_next = prog.NewContinuousVariables(num_v, "v_next")

        self.plant.SetPositions(self.context, q)
        X_WF = self.plant.CalcRelativeTransform(
            self.context, self.frame_W, self.frame_F
        )
        Jv_WF = self.plant.CalcJacobianSpatialVelocity(
            self.context,
            with_respect_to=JacobianWrtVariable.kV,
            frame_B=self.frame_F,
            p_BoBp_B=np.zeros(3),
            frame_A=self.frame_W,
            frame_E=self.frame_W,
        )
        K_X = 1.0
        V_WF_des = -K_X * se3_vector_minus(X_WF, X_WF_des)

        # add tracking objective
        # weight*|V - J*v|^2
        weight = 100.0
        prog.Add2NormSquaredCost(
            np.sqrt(weight) * Jv_WF,
            np.sqrt(weight) * V_WF_des,
            v_next,
        )

        if self.full_formulation:
            # add nullspace objectives
            # | P (v - v_des) |^2
            P = CalcNullspace(Jv_WF)
            K_null = 1.0
            v_null_des = -K_null * (q - self.q0)
            prog.Add2NormSquaredCost(
                P,
                P @ v_null_des,
                v_next,
            )

        # add min-distance constraint
        influence_distance = 0.1
        safety_distance = 0.01
        robot_clearance = self.collision_checker.CalcRobotClearance(
            q, influence_distance
        )

        dist = robot_clearance.distances()
        ddist_dq = robot_clearance.jacobians()
        if len(dist) > 0:
            dist_min = (safety_distance - dist) / self.dt
            dist_max = np.inf * dist
            prog.AddLinearConstraint(ddist_dq, dist_min, dist_max, v_next)

        if self.full_formulation:
            # add limits
            prog.AddBoundingBoxConstraint(
                (self.joint_limits.position_lower - q) / self.dt,
                (self.joint_limits.position_upper - q) / self.dt,
                v_next,
            )
            prog.AddBoundingBoxConstraint(
                self.joint_limits.velocity_lower,
                self.joint_limits.velocity_upper,
                v_next,
            )

        result = SolveOrDie(self.solver, self.solver_options, prog, tol=1e-4)
        v = result.GetSolution(v_next)
        return v
