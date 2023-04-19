"""
Utilities for basic system identification for a MultibodyPlant via inverse
dynamics.

N.B. Unless otherwise stated, inertias are expressed with respect to body
origin.

For references numbers, please see `drake_torch_sys_id.py`.

Some notation may use Drake's multibody notations:
https://drake.mit.edu/doxygen_cxx/group__multibody__quantities.html
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

from anzu.not_exported_soz.cc import AreFramesWelded
from anzu.not_exported_soz.multibody_extras import get_bodies, get_joints
from anzu.not_exported_soz.dair_pll_inertia import (
    InertialParameterConverter,
    inertia_matrix_from_vector,
    inertia_vector_from_matrix,
    parallel_axis_theorem,
)
from anzu.drake_torch_autodiff import (
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


def get_candidate_sys_id_bodies(plant):
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


def get_candidate_sys_id_joints(plant):
    joints = get_joints(plant)
    joints = [joint for joint in joints if joint.num_positions() == 1]
    return joints


def extract_inertial_param(spatial_inertia):
    mass = spatial_inertia.get_mass()
    com = spatial_inertia.get_com()
    rot_inertia = spatial_inertia.CalcRotationalInertia()
    rot_inertia = rot_inertia.CopyToFullMatrix3()
    return mass, com, rot_inertia


def get_body_inertial_param(context, body):
    spatial_inertia = body.CalcSpatialInertiaInBodyFrame(context)
    return extract_inertial_param(spatial_inertia)


def get_plant_inertial_params(plant, context, bodies):
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
    diff = np.abs(unit_inertia.CopyToFullMatrix3() - G)
    assert np.all(diff < 1e-7)
    return unit_inertia


def assemble_inertial_param(mass, com, rot_inertia):
    assert mass > 0.0
    unit_inertia = make_unit_inertia(rot_inertia / mass)
    return SpatialInertia_(mass, com, unit_inertia)


def set_body_inertial_param(context, body, mass, com, rot_inertia):
    M_BBo_B = assemble_inertial_param(mass, com, rot_inertia)
    body.SetSpatialInertiaInBodyFrame(context, M_BBo_B)


def set_plant_inertial_params(
    plant, context, bodies, masses, coms, rot_inertias
):
    tuple_iter = zip(bodies, masses, coms, rot_inertias, strict=True)
    for (body, mass, com, rot_inertia) in tuple_iter:
        set_body_inertial_param(context, body, mass, com, rot_inertia)


class InertialParameter(nn.Module):
    """
    Provides a mechanism to reparameterize spatial inertias for use in
    optimization via backprop in PyTorch.
    """

    def __init__(self, masses, coms, rot_inertias):
        # N.B. We override the constructor to ensure we can consistently
        # construct inertial parameterizations.
        super().__init__()

    def forward(self):
        """Returns masses, coms, rot_inertias."""
        raise NotImplementedError()

    @classmethod
    @torch.no_grad()
    def from_other(cls, other):
        assert isinstance(other, InertialParameter)
        masses, coms, rot_inertias = other()
        return cls(masses, coms, rot_inertias)


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


class MassAndComInertialParameter(InertialParameter):
    """
    Only vary mass and center of mass vary. This should be useful for
    identifying static components (gravity).

    Per Ale's suggestion, although this allows CoM to vary on all axes.
    """

    # Per Ale's suggestion, only let mass + com vary.
    # N.B. This lets com vary on all axes.
    def __init__(self, masses, coms, rot_inertias):
        super().__init__(masses, coms, rot_inertias)
        rot_inertias_cm = parallel_axis_theorem(
            rot_inertias, masses, coms, Ba_is_Bcm=False
        )
        unit_inertias_cm = rot_inertias_cm / masses.reshape((-1, 1, 1))
        # TODO(eric.cousineau): Should be sqrt(m) instead of log(m)?
        log_masses = torch.log(masses)
        self.log_masses = nn.Parameter(log_masses)
        self.coms = nn.Parameter(coms)
        # N.B. This is constant.
        self.unit_inertias_cm = unit_inertias_cm

    def forward(self):
        masses = torch.exp(self.log_masses)
        coms = self.coms
        rot_inertias_cm = self.unit_inertias_cm * masses.reshape((-1, 1, 1))
        rot_inertias = parallel_axis_theorem(
            rot_inertias_cm, masses, coms, Ba_is_Bcm=True
        )
        return masses, coms, rot_inertias


class UnitRotationalInertialParameter(InertialParameter):
    """
    Per Ale's suggestion, only let geometry vary by only allowing evolution of
    the rotational unit inertia (see Drake's UnitInertia) as expressed from the
    center-of-mass. Mass and center-of-mass do not vary.
    """

    def __init__(self, masses, coms, rot_inertias):
        super().__init__(masses, coms, rot_inertias)
        rot_inertias_cm = parallel_axis_theorem(
            rot_inertias, masses, coms, Ba_is_Bcm=False
        )
        unit_inertias_cm = rot_inertias_cm / masses.reshape((-1, 1, 1))
        unit_cov_cm = rot_inertia_to_cov_matrix(unit_inertias_cm)
        unit_theta_cm = matrix_to_log_cholesky_vector(unit_cov_cm)
        self.unit_theta_cm = nn.Parameter(unit_theta_cm)
        # N.B. These are constants.
        self.masses = masses
        self.coms = coms

    def forward(self):
        masses = self.masses
        coms = self.coms
        unit_cov_cm = log_cholesky_vector_to_matrix(self.unit_theta_cm)
        unit_inertias_cm = cov_matrix_to_rot_inertia(unit_cov_cm)
        rot_inertias_cm = unit_inertias_cm * masses.reshape((-1, 1, 1))
        rot_inertias = parallel_axis_theorem(
            rot_inertias_cm, masses, coms, Ba_is_Bcm=True
        )
        return masses, coms, rot_inertias


def inertial_param_to_pi(masses, coms, rot_inertias):
    # Pure torch version of
    # `InertialParameterConverter.drake_inertial_components_to_pi`
    num_bodies = masses.shape[0]
    pi = torch.zeros((num_bodies, 10))
    pi[:, 0] = masses
    pi[:, 1:4] = masses.unsqueeze(-1) * coms
    pi[:, 4:] = inertia_vector_from_matrix(rot_inertias)
    return pi


def pi_to_inertial_param(pi):
    masses = pi[:, 0]
    coms = pi[:, 1:4] / masses.unsqueeze(-1)
    rot_inertias = inertia_matrix_from_vector(pi[:, 4:])
    return masses, coms, rot_inertias


class LogCholeskyInertialParameter(InertialParameter):
    """
    Provides log-Cholesky decomposition of pseudo-inertia (expressed w.r.t.
    link origin) per [3]. This uses Matt Halm's and Bibit Bianchi's
    dair_pll_inertia code. See the corresponding doc string at the top of that
    module.
    """

    def __init__(self, masses, coms, rot_inertias):
        super().__init__(masses, coms, rot_inertias)
        pi_o = inertial_param_to_pi(masses, coms, rot_inertias)
        theta = InertialParameterConverter.pi_o_to_theta(pi_o)
        self.theta = nn.Parameter(theta)

    def forward(self):
        pi_o = InertialParameterConverter.theta_to_pi_o(self.theta)
        masses, coms, rot_inertias = pi_to_inertial_param(pi_o)
        return masses, coms, rot_inertias


class LogCholeskyLinAlgInertialParameter(InertialParameter):
    """
    Variation of `LogCholeskyInertialParameter`, using more general
    log-Cholesky decomposition based on linear algebra rather than hand-written
    symbolics.
    """

    def __init__(self, masses, coms, rot_inertias):
        super().__init__(masses, coms, rot_inertias)
        pseudo_inertias = inertial_param_to_pseudo_inertia(
            masses, coms, rot_inertias
        )
        theta = matrix_to_log_cholesky_vector(pseudo_inertias)
        self.theta = nn.Parameter(theta)

    def forward(self):
        pseudo_inertias = log_cholesky_vector_to_matrix(self.theta)
        masses, coms, rot_inertias = pseudo_inertia_to_inertial_param(
            pseudo_inertias
        )
        return masses, coms, rot_inertias


class UnitLogCholeskyInertialParameter(InertialParameter):
    """
    Only allow center of mass and geometry to vary, but keep masses fixed.
    This may be useful for identifying dynamic motions.
    """

    def __init__(self, masses, coms, rot_inertias):
        super().__init__(masses, coms, rot_inertias)
        pseudo_inertias = inertial_param_to_pseudo_inertia(
            masses, coms, rot_inertias
        )
        unit_pseudo_inertias = pseudo_inertias / masses.reshape((-1, 1, 1))
        theta = matrix_to_log_cholesky_vector(unit_pseudo_inertias)
        self.theta = nn.Parameter(theta)
        # N.B. This is constant.
        self.masses = masses

    def forward(self):
        masses = self.masses
        unit_pseudo_inertias = log_cholesky_vector_to_matrix(self.theta)
        pseudo_inertias = unit_pseudo_inertias * self.masses.reshape(
            (-1, 1, 1)
        )
        _, coms, rot_inertias = pseudo_inertia_to_inertial_param(
            pseudo_inertias
        )
        return masses, coms, rot_inertias


def torch_trace(Ms):
    """Batched trace."""
    # N.B. torch.trace() is only unbatched.
    d = torch.diagonal(Ms, dim1=-2, dim2=-1)
    return d.sum(axis=-1)


def torch_eye_repeated(B, N, *, device=None, dtype=None):
    eye = torch.eye(N, device=device, dtype=dtype)
    return eye.reshape((1, N, N)).repeat(B, 1, 1)


def torch_eye_repeated_liked(other):
    B, N, M = other.shape
    assert N == M
    return torch_eye_repeated(B, N, device=other.device, dtype=other.dtype)


def rot_inertia_to_cov_matrix(rot_inertias):
    """
    Converts rotational inertia Iₘ to covariance-parameterized rotational
    inertia, similar to [2], Eq. (13).
    """
    tr = torch_trace(rot_inertias)
    tr = tr.reshape(-1, 1, 1)
    eye = torch_eye_repeated_liked(rot_inertias)
    sigmas = 0.5 * tr * eye - rot_inertias
    return sigmas


def cov_matrix_to_rot_inertia(sigmas):
    tr = torch_trace(sigmas)
    tr = tr.reshape(-1, 1, 1)
    eye = torch_eye_repeated_liked(sigmas)
    rot_inertias = tr * eye - sigmas
    return rot_inertias


def inertial_param_to_pseudo_inertia(masses, coms, rot_inertias):
    """
    Converts inertial parameters to psuedo-inertias ℙ(4):
        P(m, c, Iₘ) = [ Σ  h ]
                      [ hᵀ m ]
    where
        Iₘ is I_BBo_B (about link origin)
        c is p_BBcm (center of mass)
        Σ = 1/2 tr(Iₘ) eye(3) - Iₘ  is the covariance
        h = m c

    Arguments:
        masses
        coms: p_BBcm for each body
        rot_inertias: I_BBo_B (about origin)
    """
    sigmas = rot_inertia_to_cov_matrix(rot_inertias)
    hs = masses.unsqueeze(-1) * coms
    N = masses.shape[0]
    pseudo_inertias = torch.zeros(
        (N, 4, 4), device=masses.device, dtype=masses.dtype
    )
    pseudo_inertias[:, :3, :3] = sigmas
    pseudo_inertias[:, :3, 3] = hs
    pseudo_inertias[:, 3, :3] = hs
    pseudo_inertias[:, 3, 3] = masses
    return pseudo_inertias


def pseudo_inertia_to_inertial_param(pseudo_inertias):
    hs = pseudo_inertias[:, :3, 3]
    masses = pseudo_inertias[:, 3, 3]
    coms = hs / masses.unsqueeze(-1)
    sigmas = pseudo_inertias[:, :3, :3]
    rot_inertias = cov_matrix_to_rot_inertia(sigmas)
    return masses, coms, rot_inertias


def to_triangular_number(N):
    """
    Compute triangular number T(N) = N * (N + 1) / 2
    https://en.wikipedia.org/wiki/Triangular_number
    """
    return int(N * (N + 1) / 2)


def from_triangular_number(count):
    """Inverse of to_triangular_number()."""
    assert count >= 0
    N = int((-1 + np.sqrt(1 + 8 * count)) / 2)
    # Ensure we have the right vector size.
    expected = to_triangular_number(N)
    assert count == expected, (count, expected)
    return N


def vec_size_to_n(vec):
    return from_triangular_number(vec.shape[-1])


def lower_triangle_to_vector(L):
    """Extracts lower triangular matrix L into a vector."""
    N = L.shape[-1]
    assert L.shape[-2:] == (N, N)
    i, j = torch.tril_indices(N, N)
    vec = L[..., i, j]
    return vec


def vector_to_lower_triangle(vec):
    """Expands vector into lower triangular matrix."""
    N = vec_size_to_n(vec)
    batch_shape = vec.shape[:-1]
    i, j = torch.tril_indices(N, N)
    L = torch.zeros(batch_shape + (N, N), device=vec.device, dtype=vec.dtype)
    L[..., i, j] = vec
    return L


def to_cholesky_vec(M):
    """Decompose matrix M = LLᵀ and extract into vector."""
    L = torch.linalg.cholesky(M)
    return lower_triangle_to_vector(L)


def from_cholesky_vec(vec):
    """Expand vector to lower triangular matrix and restore original matrix."""
    L = vector_to_lower_triangle(vec)
    M = L @ L.mT
    return M


def tril_indices_offdiag(N):
    """Get off-diagonal lower triangle indices from `torch.tril_indices()`."""
    i, j = torch.tril_indices(N, N)
    is_offdiag = i != j
    return i[is_offdiag], j[is_offdiag]


def lower_triangle_to_log_cholesky_vector(L):
    """Convert lower triangle to log-Cholesky form as in [3]."""
    # Shaping.
    N = L.shape[-1]
    count = to_triangular_number(N)
    batch_shape = L.shape[:-2]
    device = L.device
    dtype = L.dtype
    # Inidices.
    i_pen = torch.arange(N - 1)
    i_offdiag, j_offdiag = tril_indices_offdiag(N)
    # Decompose.
    exp_alpha = L[..., -1, -1]
    alpha = torch.log(exp_alpha)
    divL = L / exp_alpha.unsqueeze(-1).unsqueeze(-1)
    exp_diag_pen = divL[..., i_pen, i_pen]
    vec_offdiag = divL[..., i_offdiag, j_offdiag]
    diag_pen = torch.log(exp_diag_pen)
    vec = torch.zeros(batch_shape + (count,), device=device, dtype=dtype)
    vec[..., 0] = alpha
    vec[..., 1:N] = diag_pen
    vec[..., N:] = vec_offdiag
    return vec


def log_cholesky_vector_to_lower_triangle(vec):
    """Expands log-Cholesky to lower triangle matrix."""
    # Shaping.
    count = vec.shape[-1]
    N = from_triangular_number(count)
    batch_shape = vec.shape[:-1]
    device = vec.device
    dtype = vec.dtype
    # Inidices.
    i_pen = torch.arange(N - 1)
    i_offdiag, j_offdiag = tril_indices_offdiag(N)
    # Reconstruct.
    alpha = vec[..., 0]
    diag_pen = vec[..., 1:N]
    vec_offdiag = vec[..., N:]
    exp_alpha = torch.exp(alpha)
    exp_diag_pen = torch.exp(diag_pen)
    divL = torch.zeros(batch_shape + (N, N), device=device, dtype=dtype)
    divL[..., -1, -1] = 1.0
    divL[..., i_pen, i_pen] = exp_diag_pen
    divL[..., i_offdiag, j_offdiag] = vec_offdiag
    L = divL * exp_alpha.unsqueeze(-1).unsqueeze(-1)
    return L


def matrix_to_log_cholesky_vector(M):
    """Decomposes M into log-Cholesky vector."""
    L = torch.linalg.cholesky(M)
    vec = lower_triangle_to_log_cholesky_vector(L)
    return vec


def log_cholesky_vector_to_matrix(vec):
    """Expands log-Cholesky vector into M."""
    L = log_cholesky_vector_to_lower_triangle(vec)
    M = L @ L.mT
    return M


def inertial_param_to_pseudo_inertia_cholesky(masses, coms, rot_inertias):
    P = inertial_param_to_pseudo_inertia(masses, coms, rot_inertias)
    return to_cholesky_vec(P)


def pseudo_inertia_cholesky_to_inertial_param(vec):
    P = from_cholesky_vec(vec)
    masses, coms, rot_inertias = pseudo_inertia_to_inertial_param(P)
    return masses, coms, rot_inertias


class PseudoInertiaCholeskyInertialParameter(InertialParameter):
    """
    Using idea from [4], but rewriting it as explicit linear algebra
    operations.
    """

    def __init__(self, masses, coms, rot_inertias):
        super().__init__(masses, coms, rot_inertias)
        with torch.no_grad():
            vec = inertial_param_to_pseudo_inertia_cholesky(
                masses, coms, rot_inertias
            )
        self.vec = nn.Parameter(vec)

    def forward(self):
        return pseudo_inertia_cholesky_to_inertial_param(self.vec)


# TODO(eric.cousineau): Import something equivalent to the following:
# https://github.com/facebookresearch/differentiable-robot-model/blob/d7bd1b3b8e/differentiable_robot_model/rigid_body_params.py#L245-L250  # noqa


def inertial_param_to_log_cholesky_com(masses, coms, rot_inertias):
    """
    Matt Halm's suggestion / idea for doing
        ( log(mass)), com, matrix_to_log_cholesky_vector(sigma_cm) )
    Per his convo w/ Caleb Rucker.
    """
    rot_inertias_cm = parallel_axis_theorem(
        rot_inertias, masses, coms, Ba_is_Bcm=False
    )
    sigmas_cm = rot_inertia_to_cov_matrix(rot_inertias_cm)
    inertia_vec = matrix_to_log_cholesky_vector(sigmas_cm)
    device = masses.device
    dtype = masses.dtype
    (B,) = masses.shape
    x = torch.zeros((B, 10), device=device, dtype=dtype)
    x[..., 0] = torch.log(masses)
    x[..., 1:4] = coms
    x[..., 4:] = inertia_vec
    return x


def log_cholesky_com_to_inertial_param(x):
    masses = torch.exp(x[..., 0])
    coms = x[..., 1:4]
    sigmas_cm = log_cholesky_vector_to_matrix(x[..., 4:])
    rot_inertias_cm = cov_matrix_to_rot_inertia(sigmas_cm)
    rot_inertias = parallel_axis_theorem(
        rot_inertias_cm, masses, coms, Ba_is_Bcm=True
    )
    return masses, coms, rot_inertias


class LogCholeskyComInertialParameter(InertialParameter):
    """
    Wrapping of inertial_param_to_log_cholesky_com and its inverse.
    """

    def __init__(self, masses, coms, rot_inertias):
        super().__init__(masses, coms, rot_inertias)
        with torch.no_grad():
            vec = inertial_param_to_log_cholesky_com(
                masses, coms, rot_inertias
            )
        self.vec = nn.Parameter(vec)

    def forward(self):
        return log_cholesky_com_to_inertial_param(self.vec)


def inertial_entropic_divergence(
    pseudo_inertias_init_inv, logdet_pseudo_inertias_init, pseudo_inertias
):
    """
    Provides entropic divergence from Eq. (19) of [1]:
        d_M(ϕ, ϕ₀)² = d_F(P || P₀) = -log(|P| / |P₀|) + tr(P₀⁻¹ P) - n
    where n = 4, P ∈ ℙ(4), where ℙ(4) represents pseudo-inertias.

    Note that [1] uses Eq. (29) and removes two constant terms: log|P₀| and
    `n` (dimensionality), which won't affect gradient-based optimization.
    However, it's *very* nice to have losses that actually drive towards zero
    when possible, hence incorporating the constant terms.
    """
    # See also:
    # https://github.com/alex07143/Geometric-Robot-DynID/blob/592e64c5a9/Functions/Identification/ID_Entropic.m#L67  # noqa
    logdet_term = torch.logdet(pseudo_inertias) - logdet_pseudo_inertias_init
    trace_term = torch_trace(pseudo_inertias_init_inv @ pseudo_inertias)
    n = 4  # 4x4 pseudo-inertias
    dist_2 = -logdet_term + trace_term - n
    return dist_2


class InertialEntropicDivergence(nn.Module):
    def __init__(self, masses, coms, rot_inertias):
        super().__init__()
        # N.B. This is *not* a parameter.
        with torch.no_grad():
            pseudo_inertias_init = inertial_param_to_pseudo_inertia(
                masses, coms, rot_inertias
            )
            self.pseudo_inertias_init_inv = torch.linalg.inv(
                pseudo_inertias_init
            )
            self.logdet_pseudo_inertias_init = torch.logdet(
                pseudo_inertias_init
            )

    def forward(self, masses, coms, rot_inertias):
        psuedo_inertias = inertial_param_to_pseudo_inertia(
            masses, coms, rot_inertias
        )
        dist_2 = inertial_entropic_divergence(
            self.pseudo_inertias_init_inv,
            self.logdet_pseudo_inertias_init,
            psuedo_inertias,
        )
        return dist_2


class DrakeInverseDynamics(nn.Module):
    """
    Provides torch module for computing inverse dynamics (tau = f(q, v, vd))
    via Drake, and providing gradients for inertial parameters.

    Note: This does *no* mapping between generalized forces and actuation
    forces.
    """

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
            get_plant_inertial_params(plant_f, context_f, bodies_f),
        )
        self.inertial_params = inertial_param_cls(masses, coms, rot_inertias)

    def num_velocities(self):
        return self.plant_ad.num_velocities()

    @staticmethod
    def _to_autodiff(plant_f, bodies_f):
        plant_ad = plant_f.ToAutoDiffXd()
        bodies_ad = [plant_ad.get_body(body_f.index()) for body_f in bodies_f]
        return plant_ad, bodies_ad

    def forward(self, q, v, vd):
        masses, coms, rot_inertias = self.inertial_params()

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
            # TODO(eric.cousineau): Use multiprocessing?
            N = q.shape[0]
            tau = torch.zeros(N, self.num_velocities())
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
    set_plant_inertial_params(
        plant, context, bodies, masses, coms, rot_inertias
    )
    external_forces = MultibodyForces_(plant)
    plant.CalcForceElementsContribution(context, external_forces)
    tau = plant.CalcInverseDynamics(context, vd, external_forces)
    return tau
