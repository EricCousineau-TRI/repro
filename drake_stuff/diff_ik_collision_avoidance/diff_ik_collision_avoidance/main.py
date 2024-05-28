"""
Goal: Reproduce collision avoidance using simple kinematic integration.

Relates https://github.shared-services.aws.tri.global/robotics/anzu/issues/9463
"""

from pathlib import Path
from textwrap import dedent

import numpy as np

from pydrake.all import (
    AddMultibodyPlant,
    ApplyVisualizationConfig,
    CollisionCheckerParams,
    CommonSolverOption,
    DiagramBuilder,
    GetScopedFrameByName,
    JacobianWrtVariable,
    JointIndex,
    MathematicalProgram,
    ModelDirectives,
    MultibodyPlantConfig,
    OsqpSolver,
    PackageMap,
    Parser,
    ProcessModelDirectives,
    RigidTransform,
    RobotCollisionType,
    RobotDiagramBuilder,
    SceneGraphCollisionChecker,
    # SceneGraphConfig,
    SolverOptions,
    VisualizationConfig,
    yaml_load_typed,
)

import diff_ik_collision_avoidance as _me
from diff_ik_collision_avoidance.pose_util import se3_vector_minus


def MakeDirectives():
    raw_text = dedent(
        r"""
        directives:
        - add_model:
            name: panda
            file: package://drake_models/franka_description/urdf/panda_arm.urdf
        - add_weld:
            parent: world
            child: panda::panda_link0
        """  # noqa
    )
    directives = yaml_load_typed(schema=ModelDirectives, data=raw_text)
    return directives


def MakeMyDefaultPackageMap():
    project_dir = Path(_me.__file__).parent
    package_map = PackageMap()
    package_map.Add("diff_ik_collision_avoidance", str(project_dir))
    return package_map


def ProcessMyModelDirectives(model_directives, plant):
    parser = Parser(plant)
    parser.package_map().AddMap(MakeMyDefaultPackageMap())
    added_models = ProcessModelDirectives(
        model_directives, plant, parser=parser
    )
    return added_models


def MakeRobotDiagramBuilder(directives, time_step):
    robot_builder = RobotDiagramBuilder(time_step=time_step)
    # TODO(eric.cousineau): Use this once we have release (> 1.29.0) that has
    # SceneGraphConfig:
    # https://github.com/RobotLocomotion/drake/commit/16e6967
    # # Configure Scene Graph.
    # scene_graph_config = SceneGraphConfig()
    # scene_graph_config.default_proximity_properties.compliance_type = "compliant"  # noqa
    # robot_builder.scene_graph().set_config(scene_graph_config)
    # Load our plant.
    directives = MakeDirectives()
    ProcessMyModelDirectives(directives, robot_builder.plant())
    return robot_builder


def configuration_distance(q1, q2):
    """A boring implementation of ConfigurationDistanceFunction."""
    # TODO(eric.cousineau): Is there a way to ignore this for just a config
    # distance function?
    # What about a C++ function, without Python indirection, for better speed?
    return np.linalg.norm(q1 - q2)


def MakeCollisionChecker(robot_diagram, model_instances):
    return SceneGraphCollisionChecker(
        model=robot_diagram,
        configuration_distance_function=configuration_distance,
        edge_step_size=0.05,
        env_collision_padding=0.0,
        self_collision_padding=0.0,
        robot_model_instances=model_instances,
    )


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


def GetJoints(plant, model_instances = None):
    joints = []
    for i in range(plant.num_joints()):
        joint = plant.get_joint(JointIndex(i))
        if model_instances is None or joint.model_instance() in model_instances:
            joints.append(joint)
    return joints


def GetActiveDof(plant, model_instances):
    assert plant.num_positions() == plant.num_velocities()
    num_q = plant.num_positions()
    active_dof = np.zeros(num_q, dtype=bool)
    joints = GetJoints(plant, model_instances)
    for joint in joints:
        start = joint.position_start()
        end = start + joint.num_positions()
        active_dof[start:end] = True
    return active_dof


def GetDofJoints(plant, active_dof):
    joints = []
    for joint in GetJoints(plant):
        start = joint.position_start()
        if joint.num_positions() != 1:
            continue
        end = start + joint.num_positions()
        for i in range(start, end):
            if active_dof[i]:
                joints.append(joint)
                break
    return joints


def GetBodiesKinematicallyAffectedBy(plant, active_dof):
    joints = GetDofJoints(plant, active_dof)
    joint_indices = [joint.index() for joint in joints]
    body_indices = plant.GetBodiesKinematicallyAffectedBy(joint_indices)
    return body_indices


def IsSelfCollision(type: RobotCollisionType) -> bool:
    if type == RobotCollisionType.kSelfCollision:
        return True
    elif type == RobotCollisionType.kEnvironmentCollision:
        return False
    elif type == RobotCollisionType.kEnvironmentAndSelfCollision:
        raise RuntimeError("This case needs to be filled out")
    else:
        assert False, "Unreachable code execution"


def MaybeUseActiveDistancesAndGradients(
    active_dof,
    remove_inactive,
    active_bodies_for_collision_avoidance,
    robot_clearance,
):
    # WARNING! This is slow in Python.
    num_active_dof = np.sum(active_dof)

    all_dist = robot_clearance.distances()
    all_ddist_dq_full = robot_clearance.jacobians()
    all_collision_types = robot_clearance.collision_types()
    all_robot_indices = robot_clearance.robot_indices()
    all_other_indices = robot_clearance.other_indices()

    total = len(all_dist)
    pdist_out = np.zeros(total)
    pddist_dq_out = np.zeros((total, num_active_dof))

    if not remove_inactive:
        pdist_out[:] = all_dist
        pddist_dq_out[:] = all_ddist_dq_full[:, active_dof]
        return

    index = 0
    for all_index in range(total):
        robot_index = all_robot_indices[all_index]
        other_index = all_other_indices[all_index]
        collision_type = all_collision_types[all_index]
        is_self_collision = IsSelfCollision(collision_type)

        row_matches_active_body = (
            robot_index in active_bodies_for_collision_avoidance or
            (is_self_collision and other_index in active_bodies_for_collision_avoidance)
        )

        if row_matches_active_body:
            pdist_out[index] = all_dist[all_index]
            pddist_dq_out[index, active_dof] = all_ddist_dq_full[all_index, active_dof]
            index += 1

    return pdist_out, pddist_dq_out


class Controller:
    """
    Loosely represents the math in `MultiFramePoseStream`.

    WARNING: This is not very Systems-based.
    """

    def __init__(
        self, collision_checker, frame_F_name, dt, q0, full_formulation=False
    ):
        self.plant = collision_checker.plant()
        self.collision_checker = collision_checker
        self.dt = dt
        self.q0 = q0
        self.full_formulation = full_formulation

        self.frame_W = self.plant.world_frame()
        self.frame_F = GetScopedFrameByName(self.plant, frame_F_name)
        self.joint_limits = JointLimits(self.plant)
        self.context = self.plant.CreateDefaultContext()

        self.solver = OsqpSolver()
        self.solver_options = SolverOptions()

        num_v = self.plant.num_velocities()
        self.active_dof = np.ones(num_v, dtype=bool)
        self.active_bodies = GetBodiesKinematicallyAffectedBy(
            self.plant, self.active_dof
        )

    def step(self, q, X_WF_des):
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

        remove_inactive = True
        dist, ddist_dq = MaybeUseActiveDistancesAndGradients(
            self.active_dof,
            remove_inactive,
            self.active_bodies,
            robot_clearance,
        )
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


class Policy:
    def reset(self, X):
        raise NotImplementedError()

    def step(self, X):
        raise NotImplementedError()


class ConstantVelocityPolicy(Policy):
    """
    Creates an initial pose, then just integrates that forward in an open-loop
    fashion.
    """
    def __init__(self, v_WF, dt):
        self.dt = dt
        self.v_WF = v_WF

    def reset(self, X):
        self.X_des = X

    def step(self, X):
        X_delta = RigidTransform(self.v_WF * self.dt)
        self.X_des = X_delta @ self.X_des
        return self.X_des


class JointLimits:
    def __init__(self, plant):
        self.position_lower = plant.GetPositionLowerLimits()
        self.position_upper = plant.GetPositionUpperLimits()
        self.velocity_lower = plant.GetVelocityLowerLimits()
        self.velocity_upper = plant.GetVelocityUpperLimits()


def main():
    # Will go infeasible around 4.8s.
    t_final = 4.0
    directives = MakeDirectives()

    robot_builder = MakeRobotDiagramBuilder(directives, time_step=0.0)
    plant = robot_builder.plant()
    plant.Finalize()
    ApplyVisualizationConfig(VisualizationConfig(), robot_builder.builder())
    robot_diagram = robot_builder.Build()
    robot_diagram_context = robot_diagram.CreateDefaultContext()

    context = plant.GetMyContextFromRoot(robot_diagram_context)
    q = np.deg2rad([0.0, 0.0, 0.0, -90.0, 0.0, 90.0, 0.0])
    plant.SetPositions(context, q)
    robot_diagram.ForcedPublish(robot_diagram_context)

    dt = 0.005

    model_instances = [plant.GetModelInstanceByName("panda")]
    frame_F_name = "panda::panda_link8"

    # robot_diagram_copy = robot_diagram.Clone()  # Does not work if we add visualization.
    robot_diagram_copy = (
        MakeRobotDiagramBuilder(directives, time_step=0.0).Build()
    )
    collision_checker = MakeCollisionChecker(
        robot_diagram_copy, model_instances
    )
    controller = Controller(
        collision_checker=collision_checker,
        frame_F_name=frame_F_name,
        dt=dt,
        q0=q.copy(),
    )

    frame_W = plant.world_frame()
    frame_F = GetScopedFrameByName(plant, frame_F_name)
    X = plant.CalcRelativeTransform(context, frame_W, frame_F)

    policy = ConstantVelocityPolicy(v_WF=np.array([-0.2, 0.01, -0.2]), dt=dt)
    policy.reset(X)

    t_rel = 0.0
    try:
        while t_rel < t_final:
            X = plant.CalcRelativeTransform(context, frame_W, frame_F)
            X_des = policy.step(X)
            v = controller.step(q, X_des)

            # Update positions.
            q += v * dt
            plant.SetPositions(context, q)
            robot_diagram.ForcedPublish(robot_diagram_context)

            # Advance to next time step.
            t_rel += dt

    except RuntimeError as e:
        print(f"Failure at t={t_rel}")
        print(e)
        print(f"q: {q.tolist()}")

    print()
    return t_rel


if __name__ == "__main__":
    main()
