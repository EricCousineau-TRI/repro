"""
Utilities for basic system identification for a MultibodyPlant via inverse
dynamics.

N.B. Unless otherwise stated, inertias are expressed with respect to body
origin.
"""

import numpy as np
import torch
from torch import nn

from pydrake.autodiffutils import AutoDiffXd
from pydrake.multibody.plant import MultibodyPlant_
from pydrake.multibody.tree import (
    MultibodyForces_,
    SpatialInertia_,
    UnitInertia_,
)

from dair_pll.inertia import (
    InertialParameterConverter,
    inertia_matrix_from_vector,
    inertia_vector_from_matrix,
)

from my_package.not_exported_soz import AreFramesWelded, get_bodies
from my_package.drake_torch_autodiff import (
    drake_torch_function,
)


def get_welded_subgraphs(plant):
    bodies = get_bodies(plant)
    bodies_seen = set()
    subgraphs = []
    for body in bodies:
        subgraph = plant.GetBodiesWeldedTo(body)
        if body not in bodies_seen:
            subgraphs.append(subgraph)
        bodies_seen |= set(subgraph)
    return subgraphs


def get_identifiable_bodies(plant):
    subgraphs = get_welded_subgraphs(plant)
    bodies = []
    for subgraph in subgraphs:
        first_body = subgraph[0]
        # Ignore world subgraph.
        if AreFramesWelded(
            plant, first_body.body_frame(), plant.world_frame()
        ):
            continue
        masses = [body.default_mass() for body in subgraph]
        if sum(masses) == 0.0:
            continue
        if len(subgraph) == 1:
            (body,) = subgraph
        else:
            # Take body with greatest mass and optimize it.
            # TODO(eric.cousineau): Should instead distribute more evenly?
            i = np.argmax(masses)
            body = subgraph[i]
        bodies.append(body)
    return bodies


def extract_inertial_param(spatial_inertia):
    mass = spatial_inertia.get_mass()
    com = spatial_inertia.get_com()
    rot_inertia = spatial_inertia.CalcRotationalInertia()
    rot_inertia = rot_inertia.CopyToFullMatrix3()
    return mass, com, rot_inertia


def get_body_inertial_param(context, body):
    spatial_inertia = body.CalcSpatialInertiaInBodyFrame(context)
    return extract_inertial_param(spatial_inertia)


def get_plant_inertial_param(plant, context, bodies):
    masses = []
    coms = []
    rot_inertias = []
    for body in bodies:
        mass, com, rot_inertia = get_body_inertial_param(context, body)
        masses.append(mass)
        coms.append(com)
        rot_inertias.append(rot_inertia)
    masses, coms, rot_inertias = map(np.asarray, (masses, coms, rot_inertias))
    return masses, coms, rot_inertias


def make_unit_inertia(G):
    unit_inertia = UnitInertia_(
        Ixx=G[0, 0],
        Iyy=G[1, 1],
        Izz=G[2, 2],
        Ixy=G[0, 1],
        Ixz=G[0, 2],
        Iyz=G[1, 2],
    )
    assert np.all(unit_inertia.CopyToFullMatrix3() == G)
    return unit_inertia


def assemble_inertial_param(mass, com, rot_inertia):
    assert mass > 0.0
    unit_inertia = make_unit_inertia(rot_inertia / mass)
    return SpatialInertia_(mass, com, unit_inertia)


def set_body_inertial_param(context, body, mass, com, rot_inertia):
    M_BBo_B = assemble_inertial_param(mass, com, rot_inertia)
    body.SetSpatialInertiaInBodyFrame(context, M_BBo_B)


def set_plant_inertial_param(
    plant, context, bodies, masses, coms, rot_inertias
):
    tuple_iter = zip(bodies, masses, coms, rot_inertias, strict=True)
    for (body, mass, com, rot_inertia) in tuple_iter:
        set_body_inertial_param(context, body, mass, com, rot_inertia)


class InertialParameter(nn.Module):
    def __init__(self, masses, coms, rot_inertias):
        # N.B. We override the constructor to ensure we can consistently
        # construct inertial parameterizations.
        super().__init__()

    def forward(self):
        """Returns masses, coms, rot_inertias."""
        raise NotImplementedError()


class RawInertialParameter(InertialParameter):
    def __init__(self, masses, coms, rot_inertias):
        super().__init__(masses, coms, rot_inertias)
        self.masses = nn.Parameter(masses)
        self.coms = nn.Parameter(coms)
        self.rot_inertias = nn.Parameter(rot_inertias)

    def forward(self):
        return self.masses, self.coms, self.rot_inertias


class VectorInertialParameter(InertialParameter):
    def __init__(self, masses, coms, rot_inertias):
        super().__init__(masses, coms, rot_inertias)
        pi_o = inertial_param_to_pi(masses, coms, rot_inertias)
        self.pi_o = nn.Parameter(pi_o)

    def forward(self):
        return pi_to_inertial_param(self.pi_o)


def inertial_param_to_pi(masses, coms, rot_inertias):
    num_bodies = masses.shape[0]
    pi = torch.zeros((num_bodies, 10))
    pi[:, 0] = masses
    pi[:, 1:4] = coms
    pi[:, 4:] = inertia_vector_from_matrix(rot_inertias)
    return pi


def pi_to_inertial_param(pi):
    masses = pi[:, 0]
    coms = pi[:, 1:4]
    rot_inertias = inertia_matrix_from_vector(pi[:, 4:])
    return masses, coms, rot_inertias


class LogCholeskyInertialParameter(InertialParameter):
    # See dair_pll.inertia doc string.
    def __init__(self, masses, coms, rot_inertias):
        super().__init__(masses, coms, rot_inertias)
        pi_o = inertial_param_to_pi(masses, coms, rot_inertias)
        theta = InertialParameterConverter.pi_o_to_theta(pi_o)
        self.theta = nn.Parameter(theta)

    def forward(self):
        pi_o = InertialParameterConverter.theta_to_pi_o(self.theta)
        masses, coms, rot_inertias = pi_to_inertial_param(pi_o)
        return masses, coms, rot_inertias


class DrakeLearnableModel(nn.Module):
    def __init__(
        self,
        plant_f,
        bodies_f,
        inertial_param_cls=RawInertialParameter,
    ):
        super().__init__()
        # TODO(eric.cousineau): How to handle bodies that are welded? Record
        # initial ratio of mass / CoM / inertia?
        context_f = plant_f.CreateDefaultContext()
        plant_ad, bodies_ad = self._to_autodiff(plant_f, bodies_f)
        self.plant_ad = plant_ad
        self.bodies_ad = bodies_ad
        self.context_ad = self.plant_ad.CreateDefaultContext()
        masses, coms, rot_inertias = map(
            torch.from_numpy,
            get_plant_inertial_param(plant_f, context_f, bodies_f),
        )
        self.inertial = inertial_param_cls(masses, coms, rot_inertias)
        self.num_u = self.plant_ad.num_actuators()

    @staticmethod
    def _to_autodiff(plant_f, bodies_f):
        plant_ad = plant_f.ToAutoDiffXd()
        bodies_ad = [plant_ad.get_body(body_f.index()) for body_f in bodies_f]
        return plant_ad, bodies_ad

    def forward(self, q, v, vd):
        masses, coms, rot_inertias = self.inertial()

        def forward_unbatched(q_i, v_i, vd_i):
            return plant_inverse_dynamics.torch(
                self.plant_ad,
                self.context_ad,
                q_i,
                v_i,
                vd_i,
                self.bodies_ad,
                masses,
                coms,
                rot_inertias,
            )

        if q.dim() == 2:
            # Batched.
            N = q.shape[0]
            tau = torch.zeros(N, self.num_u)
            for i in range(N):
                tau[i] = forward_unbatched(q[i], v[i], vd[i])
        elif q.dim() == 1:
            tau = forward_unbatched(q, v, vd)
        else:
            assert False
        return tau


@drake_torch_function
def plant_inverse_dynamics(
    plant, context, q, v, vd, bodies, masses, coms, rot_inertias
):
    plant.SetPositions(context, q)
    plant.SetVelocities(context, v)
    set_plant_inertial_param(
        plant, context, bodies, masses, coms, rot_inertias
    )
    external_forces = MultibodyForces_(plant)
    plant.CalcForceElementsContribution(context, external_forces)
    tau = plant.CalcInverseDynamics(context, vd, external_forces)
    return tau


"""
class DrakeInverseDynamicsFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        plant,
        context,
        q,
        v,
        vd,
        bodies,
        masses,
        coms,
        # Must be body origin, not center of mass.
        rot_inertias,
    ):
        # Torch to numpy.
        q, v, vd = map(torch.Tensor.numpy, (q, v, vd))
        # Torch w/ grad to AutoDiff.
        (masses, coms, rot_inertias), shapes = (
            torch_to_initialize_autodiff_list([masses, coms, rot_inertias])
        )
        # Compute via AutoDiff.
        tau = plant_inverse_dynamics(
            plant, context, q, v, vd, bodies, masses, coms, rot_inertias
        )
        # Convert back.
        tau, dtau_dw = autodiff_to_torch_and_grad(tau)
        # Remember local gradient and shapes.
        ctx.save_for_backward(dtau_dw)
        ctx.shapes = shapes
        return tau

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dL_dtau):
        # Goal: Return dL_dw = dL/dw
        shapes = ctx.shapes
        dtau_dw, = ctx.saved_tensors
        # Chain rule.
        dL_dw = dL_dtau @ dtau_dw
        # Reshape gradients, dL_dw -> grad_w
        grad_masses, grad_coms, grad_rot_inertias = unflatten_list(
            dL_dw, shapes
        )
        grad_plant = None
        grad_context = None
        grad_q = None
        grad_v = None
        grad_vd = None
        grad_bodies = None
        # Match same order as arguments to forward().
        return (
            grad_plant,
            grad_context,
            grad_q,
            grad_v,
            grad_vd,
            grad_bodies,
            grad_masses,
            grad_coms,
            grad_rot_inertias,
        )
"""
