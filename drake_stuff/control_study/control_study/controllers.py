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
        self.state_input = self.DeclareVectorInputPort("state", self.num_q)
        self.inputs_motion_desired = declare_spatial_motion_inputs(
            self,
            name_X="X_des",
            name_V="V_des",
            name_A="A_des",
        )
        self.torques_output = self.DeclareVectorOutputPort(
            "torques",
            size=self.q,
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
    # For now, just same value, with critically damped.
    kp_t: float
    kp_q: float

    def calc(self, num_q):
        num_spatial = 6
        ones_t = np.ones(num_spatial)
        kp_t = ones_t * self.kp_t
        kd_t = 2 * np.sqrt(kp_t)
        ones_q = np.ones(num_q)
        kp_q = ones_q * self.kp_q
        kd_q = 2 * np.sqrt(kp_q)
        return kp_t, kd_t, kp_q, kd_q


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
        self.kp_t, self.kd_t, self.kp_q, self.kd_q = gains.calc(self.num_q)

    def calc_control(self, pose_actual, pose_desired, q0):
        M, C, tau_g = calc_dynamics(self.plant, self.context)
        Minv = inv(M)

        # Compute spatial feedback.
        X, V, Jt, Jtdot_v = pose_actual
        X_des, V_des, A_des = pose_desired
        e = se3_vector_minus(X, X_des)
        ed = (V - V_des).get_coeffs()
        edd_c = A_des.get_coeffs() - self.kp_t * e - self.kd_t * ed
        Mt, _, Jt, _, Nt_T = reproject_mass(Minv, Jt)
        Ft = Mt @ (edd_c - Jtdot_v)

        # Compute posture feedback.
        q = self.plant.GetPositions(self.context)
        v = self.plant.GetVelocities(self.context)
        e = q - q0
        ed = v
        edd_c = -self.kp_q * e - self.kd_q * ed
        Fp = M @ edd_c

        # Sum up tasks and cancel gravity + Coriolis terms.
        tau = Jt.T @ Ft + Nt_T @ Fp + C - tau_g
        return tau
