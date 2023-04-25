import dataclasses as dc
import functools

import numpy as np

from pydrake.all import (
    DiagramBuilder,
    Integrator,
    PiecewisePolynomial,
    Simulator,
)
from pydrake.common.eigen_geometry import Quaternion, Quaternion_
from pydrake.common.value import Value
from pydrake.math import (
    RigidTransform,
    RollPitchYaw_,
    RotationMatrix,
    RotationMatrix_,
)
from pydrake.multibody.math import SpatialAcceleration, SpatialVelocity
from pydrake.multibody.tree import SpatialInertia, UnitInertia
from pydrake.symbolic import Evaluate, Expression, Jacobian, Variable
from pydrake.systems.framework import BasicVector, LeafSystem
from pydrake.systems.primitives import Adder


# Er, misc / bad sorting.


def normalize(x, *, tol=1e-10):
    x = np.asarray(x)
    n = np.linalg.norm(x)
    assert n >= tol
    return x / n


def maxabs(x):
    return np.max(np.abs(x))


def cat(*args):
    """Short-hand to concatenates arguments."""
    return np.concatenate(args)


def cross(a, b):
    return np.cross(a, b)


# Rotation coordinates.


@dc.dataclass
class RotationInfo:
    num_rot: int
    # Unity representation (for offsetting, integrator, etc.).
    r0: np.ndarray
    # q(R)
    from_rotation: object
    # R(r), w(r, rd), a(r, rd, rdd)
    calc_values: object
    # qd = calc_rate(r, w)
    calc_rate: object
    # Projects "raw" inputs (r_i, rd_i, rdd_i) onto proper subspace
    # (r, rd, rdd)
    project_values: object


@functools.lru_cache
def make_rot_info_quat():
    def from_rotation(R):
        return R.ToQuaternion().wxyz()

    def calc_values(q, qd, qdd):
        R = RotationMatrix(Quaternion(wxyz=q))
        # Compute world-frame angular velocity and acceleration according
        # to Paul's book, Advanced Dynamics & Motion Simulation, Eq. 9.3.5
        # (p. 77).
        s = q[0]
        sd = qd[0]
        v = q[1:]
        vd = qd[1:]
        w = 2 * (s * vd - sd * v + cross(v, vd))
        sdd = qdd[0]
        vdd = qdd[1:]
        a = 2 * (s * vdd - sdd * v + cross(vd, vd) + cross(v, vdd))
        return R, w, a

    def project_values(r, rd, rdd):
        rt = (r, rd, rdd)
        qt = normalize_2nd(rt)
        return qt

    def calc_rate(q, w):
        # Compute time-derivative of quaternion coordinates using similar
        # equations as above.
        s = q[0]
        v = q[1:]
        sd = -v.dot(w) / 2
        vd = (s * w - cross(v, w)) / 2
        # TODO(eric.cousineau): Do we need chain rule for norm like what
        # Drake's `QuaternionRateToAngularVelocityMatrix` does? Tests seem to
        # pass without it.
        qd = np.zeros(4)
        qd[0] = sd
        qd[1:] = vd
        return qd

    return RotationInfo(
        num_rot=4,
        r0=np.array([1.0, 0.0, 0.0, 0.0]),
        from_rotation=from_rotation,
        calc_values=calc_values,
        project_values=project_values,
        calc_rate=calc_rate,
    )


def make_spline_trajectory(ts, qs, *, return_spline=False):
    """
    Creates a cubic spline that is C2.

    Returns:
        (func, q_spline), where `func(t) = q, v, vd`.
    """
    assert ts[0] == 0.0
    qs = np.asarray(qs)
    q_spline = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
        breaks=ts,
        samples=qs.T,
        periodic_end_condition=True,
    )
    v_spline = q_spline.derivative()
    vd_spline = v_spline.derivative()

    def func(t):
        q = q_spline.value(t).reshape((-1,))
        v = v_spline.value(t).reshape((-1,))
        vd = vd_spline.value(t).reshape((-1,))
        return q, v, vd

    if return_spline:
        return func, q_spline
    else:
        return func


def quat_unwrap(q, *, q_prev):
    dot = q.dot(q_prev)
    if dot < 0:
        q *= -1
    return q


def quat_unwrap_traj(qs):
    qs_out = np.zeros_like(qs)
    count = len(qs)
    qs_out[0] = qs[0]
    for i in range(1, count):
        qs_out[i] = quat_unwrap(qs[i], q_prev=qs[i - 1])
    return qs_out


def make_so3_spline_trajectory_via_projection(ts, Rs):
    rot_info = make_rot_info_quat()
    qs = [rot_info.from_rotation(R) for R in Rs]
    qs = quat_unwrap_traj(qs)
    q_func = make_spline_trajectory(ts, qs)
    return make_rot_and_proj_func(rot_info, q_func)


def make_se3_spline_trajectory(ts, Xs):
    Rs = [X.rotation() for X in Xs]
    ps = [X.translation() for X in Xs]
    R_func = make_so3_spline_trajectory_via_projection(ts, Rs)
    p_func = make_spline_trajectory(ts, ps)

    def func(t):
        Xt = cat_se3_2nd(R_func(t), p_func(t))
        return Xt

    return func


def make_rot_proj_func(rot_info, func):
    def wrap(t):
        f, fd, fdd = func(t)
        r, rd, rdd = rot_info.project_values(f, fd, fdd)
        return r, rd, rdd

    return wrap


def make_rot_calc_func(rot_info, func):
    def wrap(t):
        r, rd, rdd = func(t)
        return rot_info.calc_values(r, rd, rdd)

    return wrap


def make_rot_and_proj_func(rot_info, func):
    proj = make_rot_proj_func(rot_info, func)
    calc = make_rot_calc_func(rot_info, proj)
    return calc


# Reference trajectories, 2nd-order stuff.

# TODO(eric.cousineau): Should use Drake's SE(3) trajectory / polynomial
# classes, and/or use AutoDiff.


def cat_se3_2nd(Rt, pt):
    R, w, wd = Rt
    p, pd, pdd = pt
    X = RigidTransform(R, p)
    V = SpatialVelocity(w, pd)
    A = SpatialAcceleration(wd, pdd)
    return (X, V, A)


def normalize_2nd(rt, *, tol=1e-10):
    r, rd, rdd = rt
    # Let s = norm(r), s2 = s*s
    s2 = r.dot(r)
    assert s2 > 1e-10
    s = np.sqrt(s2)
    q = r / s
    # First derivative.
    sd = q.dot(rd)
    a = rd * s - r * sd
    b = s2
    qd = a / b
    # Second derivative.
    sdd = qd.dot(rd) + q.dot(rdd)
    ad = rdd * s - r * sdd
    bd = 2 * s * sd
    qdd = (ad * b - a * bd) / (b * b)
    return (q, qd, qdd)
