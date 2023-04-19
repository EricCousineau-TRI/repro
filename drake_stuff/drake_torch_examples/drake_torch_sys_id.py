"""
Composition of torch and Drake functionality to do robust(ish) system id on
articulated rigid body systems.

Some References:

[1]: Lee, Taeyoon, Patrick M. Wensing, and Frank C. Park. "Geometric
     Robot Dynamic Identification: A Convex Programming Approach." IEEE
     Transactions on Robotics (2020). https://doi.org/10.1109/TRO.2019.2926491.

[2]: Sutanto, Giovanni, Austin S Wang, Yixin Lin, Mustafa Mukadam, Gaurav S
     Sukhatme, Akshara Rai, and Franziska Meier. "Encoding Physical Constraints
     in Differentiable Newton-Euler Algorithm," (2020).
     https://arxiv.org/abs/2001.08861

[3]: Rucker, Caleb, and Patrick M. Wensing. "Smooth Parameterization of
     Rigid-Body Inertia." IEEE Robotics and Automation Letters (April 2022):
     https://doi.org/10.1109/LRA.2022.3144517

[4]: Reuss, Moritz, Niels van Duijkeren, Robert Krug, Philipp Becker, Vaisakh
     Shaj, and Gerhard Neumann. "End-to-End Learning of Hybrid Inverse Dynamics
     Models for Precise and Compliant Impedance Control." (2022).
     https://doi.org/10.48550/arXiv.2205.13804.
"""

from functools import partial

import numpy as np
import torch
from torch import nn

from anzu.not_exported_soz.containers import take_first
from anzu.not_exported_soz.cc import (
    ApplyBodyParam,
    ApplyJointParam,
    ExtractBodyParam,
    ExtractJointParam,
    ModelParam,
)
from anzu.drake_torch_dynamics import (
    DrakeInverseDynamics,
    InertialEntropicDivergence,
    InertialParameter,
    LogCholeskyInertialParameter,
    set_plant_inertial_params,
)
from anzu.friction import (
    calc_joint_dry_friction,
    regularizer_tanh,
)
from anzu.friction_fit import (
    JointFriction,
    PositiveScalar,
)


class DynamicsModel(nn.Module):
    """
    Given the following equations of motion, roughly following Eq. (43) of [1]:
        τ_id = (Mₙ(q) + Jᵣ)*vd + Hₙ(q, v) = τ_cmd + τ_friction + τ_unmodelled
    where
        Mₙ(q) is the *nominal* mass matrix (no rotor inertias)
        Jᵣ represents the diagonalized reflected rotor inertias
        Hₙ(q, v) are the *nominal* bias terms, H(q, v) = C(q, v) + g(q), with
            no viscous (damping) or dry friction terms
        τ_cmd are the commanded / desired torques
        τ_friction are the dry and viscous joint frictions
        τ_unmodelled are unmodeled torques (e.g. external contacts, noise)

    This computes and returns `τ_cmd` as follows:
        τ_cmd = τ_id - τ_friction
    This can be then be used do simple non-linear least squares estimation via
    PyTorch + backprop to provide a better model *for control*.

    Note:
        While it may seem more ideal to use τ_measured from the robot driver,
        on the Panda arm, it is vastly different than τ_cmd. Given that our
        goal is to use this for better control (rather than actual torque
        estimation), we use τ_cmd.
    """

    @classmethod
    def from_plant(
        cls, plant, bodies, inertial_param_cls=LogCholeskyInertialParameter
    ):
        inverse_dynamics = DrakeInverseDynamics(
            plant,
            bodies,
            inertial_param_cls=inertial_param_cls,
        )
        # TODO(eric.cousineau): Assert no joint damping? Or initialize off of
        # it?
        return cls(inverse_dynamics)

    def __init__(self, inverse_dynamics):
        super().__init__()
        num_v = inverse_dynamics.num_velocities()
        self.inverse_dynamics = inverse_dynamics
        sample_param = take_first(self.inverse_dynamics.parameters())
        self.joint_friction = JointFriction.make_initial_guess(num_v)
        self.joint_friction.to(sample_param)
        # gear_ratio^2 * rotor_inertia
        self.rotor_reflected_inertia = PositiveScalar(1e-8 * torch.ones(num_v))
        self.rotor_reflected_inertia.to(sample_param)

    @property
    def inertial_params(self):
        return self.inverse_dynamics.inertial_params

    def reparameterize_inertial(self, inertial_param_cls):
        new = inertial_param_cls.from_other(self.inertial_params)
        self.inverse_dynamics.inertial_params = new

    def forward(self, q, v, vd, *, static=False):
        tau_id_nominal = self.inverse_dynamics(q, v, vd)
        if static:
            # We should only have gravitational terms. Explicitly remove
            # auxiliary terms (friction, rotor inretia).
            assert v.abs().max() < 1e-6
            assert vd.abs().max() < 1e-6
            return tau_id_nominal
        tau_id_rotor = self.rotor_reflected_inertia() * vd
        tau_id = tau_id_nominal + tau_id_rotor
        tau_friction = self.joint_friction(v)
        tau_cmd = tau_id - tau_friction
        return tau_cmd


class DynamicsModelTrajectoryLoss(nn.Module):
    def __init__(self, model, gamma, tau_loss=nn.functional.mse_loss):
        super().__init__()
        self.model = model
        masses, coms, rot_inertias = self.model.inertial_params()
        self.gamma = gamma
        self.regularizer = InertialEntropicDivergence(
            masses, coms, rot_inertias
        )
        self.tau_loss = tau_loss

    def forward(self, q, v, vd, tau_cmd_target, *, static=False):
        tau_cmd_est = self.model(q, v, vd, static=static)
        tau_loss = self.tau_loss(tau_cmd_target, tau_cmd_est)
        mass, coms, rot_inertias = self.model.inertial_params()
        # N.B. For nominal sys id, this will be batched according to number of
        # bodies.
        regularizer_loss = (
            self.gamma * self.regularizer(mass, coms, rot_inertias).mean()
        )
        # TODO(eric.cousineau): Regularize other parameters (friction)?
        loss = tau_loss + regularizer_loss
        loss_dict = {
            "tau": tau_loss.detach().item(),
            "regularizer": regularizer_loss.detach().item(),
        }
        return loss, loss_dict


def extract_model_param_from_plant(plant, context, bodies, joints):
    body_map = {}
    joint_map = {}
    # Extract body param.
    for body in bodies:
        body_param = ExtractBodyParam(context, body)
        body_map[body.name()] = body_param
    # Extract joint param.
    for j, joint in enumerate(joints):
        joint_param = ExtractJointParam(context, joint)
        joint_map[joint.name()] = joint_param
    model_param = ModelParam(joint_param=joint_map, body_param=body_map)
    return model_param


@torch.no_grad()
def extract_model_param_from_torch(plant, bodies, joints, dyn_model):
    # Extract body and joint parameters from torch model.
    # - Body.
    masses, coms, rot_inertias = dyn_model.inertial_params()
    masses = masses.numpy()
    coms = coms.numpy()
    rot_inertias = rot_inertias.numpy()
    # - Joint.
    reflected_rotor_inertia = dyn_model.rotor_reflected_inertia().numpy()
    dry_v0 = dyn_model.joint_friction.dry.v0().numpy()
    dry_u_max = dyn_model.joint_friction.dry.u_max().numpy()
    viscous_b = dyn_model.joint_friction.b().numpy()
    # Update plant.
    context = plant.CreateDefaultContext()
    set_plant_inertial_params(
        plant, context, bodies, masses, coms, rot_inertias
    )
    # Extract body and (baseline) joint parameters.
    model_param = extract_model_param_from_plant(
        plant, context, bodies, joints
    )
    # Map in extra joint parameters.
    joint_param_iter = model_param.joint_param.values()
    for j, joint_param in enumerate(joint_param_iter):
        joint_param.reflected_rotor_inertia = reflected_rotor_inertia[j]
        dry_param = joint_param.friction.dry
        dry_param.max_generalized_force = dry_u_max[j]
        dry_param.v0 = dry_v0[j]
        viscous_param = joint_param.friction.viscous
        viscous_param.damping = viscous_b[j]
    return model_param


def apply_model_param_to_plant(plant, context, bodies, joints, model_param):
    joint_iter = zip(joints, model_param.joint_param.items())
    joint_params = []
    for joint, (joint_name, joint_param) in joint_iter:
        assert joint.name() == joint_name
        # TODO(eric.cousineau): Reenable this. Currently, a no-op.
        # ApplyJointParam(joint_param, context, joint, include_damping=False)
        joint_params.append(joint_param)
    body_iter = zip(bodies, model_param.body_param.items())
    for body, (body_name, body_param) in body_iter:
        assert body.name() == body_name
        ApplyBodyParam(body_param, context, body)
    return joint_params


def expand_joint_params(joint_params):
    # Make it easier to do math.
    num_v = len(joint_params)
    viscous_b = np.zeros(num_v)
    dry_v0 = np.zeros(num_v)
    dry_u_max = np.zeros(num_v)
    reflected_rotor_inertias = np.zeros(num_v)
    for j, param in enumerate(joint_params):
        viscous_b[j] = param.friction.viscous.damping
        dry_v0[j] = param.friction.dry.v0
        dry_u_max[j] = param.friction.dry.max_generalized_force
        reflected_rotor_inertias[j] = param.reflected_rotor_inertia
    return viscous_b, dry_v0, dry_u_max, reflected_rotor_inertias


def joint_params_to_calc_friction(joint_params):
    (
        viscous_b,
        dry_v0,
        dry_u_max,
        reflected_rotor_inertias,
    ) = expand_joint_params(joint_params)
    regularizer = partial(regularizer_tanh, m=None)

    def calc_friction(v):
        u_dry = calc_joint_dry_friction(
            v,
            v0=dry_v0,
            u_max=dry_u_max,
            regularizer=regularizer,
        )
        u_viscous = -viscous_b * v
        return u_dry + u_viscous

    return calc_friction, reflected_rotor_inertias
