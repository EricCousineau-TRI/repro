from textwrap import dedent

import numpy as np

from pydrake.all import (
    ApplyVisualizationConfig,
    GetScopedFrameByName,
    ModelDirectives,
    RigidTransform,
    VisualizationConfig,
    yaml_load_typed,
)

from diff_ik_collision_avoidance.basics import (
    MakeCollisionChecker,
    MakeRobotDiagramBuilder,
)
from diff_ik_collision_avoidance.diff_ik import DiffIk


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
    frame_W = plant.world_frame()
    frame_F = GetScopedFrameByName(plant, frame_F_name)

    # robot_diagram_copy = robot_diagram.Clone()  # Does not work if we add visualization.
    robot_diagram_copy = (
        MakeRobotDiagramBuilder(directives, time_step=0.0).Build()
    )
    collision_checker = MakeCollisionChecker(
        robot_diagram_copy, model_instances
    )
    diff_ik = DiffIk(
        collision_checker=collision_checker,
        # N.B. We have to get the collision-checker's version of frame_F.
        frame_F=robot_diagram_copy.plant().get_frame(frame_F.index()),
        dt=dt,
        q0=q.copy(),
    )

    X_WF = plant.CalcRelativeTransform(context, frame_W, frame_F)
    policy = ConstantVelocityPolicy(v_WF=np.array([-0.2, 0.01, -0.2]), dt=dt)
    policy.reset(X_WF)

    # This is similar to `DifferentialInverseKinematicsIntegrator` in that we
    # integrator our desired velocities into a start desired configuration, so
    # it is effectively open-loop.
    t_rel = 0.0
    q_desired = q.copy()
    try:
        while t_rel < t_final:
            X_WF = plant.CalcRelativeTransform(context, frame_W, frame_F)
            X_WF_des = policy.step(X_WF)
            v_desired = diff_ik.solve(q_desired, X_WF_des)

            # Update positions by doing simple first-order Euler integration.
            q_desired += v_desired * dt
            # Display *desired* configuration.
            # This is *not* a simulated plant.
            plant.SetPositions(context, q_desired)
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
