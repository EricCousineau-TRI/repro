# Extracted from:
# https://github.com/RobotLocomotion/drake/pull/18815

import dataclasses as dc
import functools

import numpy as np

from pydrake.all import (
    Evaluate,
    Expression,
    Jacobian,
    Quaternion_,
    RollPitchYaw_,
    RotationMatrix_,
    Variable,
)
import pydrake.math as drake_math

from so2_so3_helpers import skew, unskew, cat

# Symbolics.


@np.vectorize
def to_float(x):
    if isinstance(x, float):
        return x
    elif isinstance(x, Expression):
        _, (c,) = x.Unapply()
        assert isinstance(c, float), type(c)
        return c
    else:
        assert False


def drake_sym_replace(expr, old, new):
    def recurse(sub):
        return drake_sym_replace(sub, old, new)

    if isinstance(expr, np.ndarray):
        recurse = np.vectorize(recurse)
        return recurse(expr)

    if not isinstance(expr, Expression):
        return expr
    if expr.EqualTo(old):
        return new
    ctor, old_args = expr.Unapply()
    new_args = [recurse(x) for x in old_args]
    return ctor(*new_args)


def differentiate(A, x):
    dx = np.vectorize(lambda ai: ai.Differentiate(x))
    return dx(A)


def derivatives_1st(A, xs, xds):
    Ad = np.zeros_like(A)
    for x, xd in zip(xs, xds):
        J = differentiate(A, x)
        Ad += J * xd
    return Ad


def derivatives_1st_and_2nd(A, xs, xds, xdds):
    Ad = derivatives_1st(A, xs, xds)
    vs = cat(xs, xds)
    vds = cat(xds, xdds)
    Add = derivatives_1st(Ad, vs, vds)
    return Ad, Add


def sym_values_and_1st_and_2nd_derivatives(names):
    x = [Variable(name) for name in names]
    xd = [Variable(f"{name}_dot") for name in names]
    xdd = [Variable(f"{name}_ddot") for name in names]
    return [np.array(s) for s in (x, xd, xdd)]


def aliased_scalar(x, i):
    return x[i : i + 1].reshape(())


def make_env(syms, xs=None):
    return_xs = False
    if xs is None:
        alias = True
        xs = tuple(np.zeros(len(sym_i)) for sym_i in syms)
        return_xs = True
    else:
        alias = False
    env = {}
    for sym, x in zip(syms, xs):
        for i, sym_i in enumerate(sym):
            if alias:
                xi = aliased_scalar(x, i)
            else:
                xi = x[i]
            env[sym_i] = xi
    if return_xs:
        return env, xs
    else:
        return env


# Rotation coordinates.


@dc.dataclass
class RotationInfo:
    num_rot: int
    # Unity representation (for offsetting, integrator, etc.).
    r0: np.ndarray
    # R(r), w(r, rd), a(r, rd, rdd)
    calc_values: object
    # J+ = dr/dw
    calc_rate_jacobian: object
    # J = dw/dr
    calc_angular_velocity_jacobian: object
    # Projects "raw" inputs (r_i, rd_i, rdd_i) onto proper subspace
    # (r, rd, rdd)
    project_values: object


def evaluate_sym(expr, env):
    for old, new in env.items():
        expr = drake_sym_replace(expr, old, new)
    return expr


def infer_sym_dtype_stuff(x):
    if x.dtype == object:
        T = Expression
        evaluate = evaluate_sym
        tol = None
    else:
        T = float
        evaluate = Evaluate
        tol = 1e-10
    return T, evaluate, tol


@functools.lru_cache
def make_rot_info_rpy_sym():
    """Makes RollPitchYaw rotation coordinate info."""
    syms = sym_values_and_1st_and_2nd_derivatives("rpy")
    r_s, rd_s, rdd_s = syms

    # Compute expressions.
    R_s = RollPitchYaw_[Expression](r_s).ToRotationMatrix().matrix()
    Rd_s, Rdd_s = derivatives_1st_and_2nd(R_s, r_s, rd_s, rdd_s)
    wh_s = Rd_s @ R_s.T
    ah_s = wh_s.T @ wh_s + Rdd_s @ R_s.T
    # ah = derivatives_1st(wh, cat(r, rd), cat(rd, rdd))
    w_s = unskew(wh_s, tol=None)
    J_s = Jacobian(w_s, rd_s)

    def calc_values(r_e, rd_e, rdd_e):
        T, evaluate, tol = infer_sym_dtype_stuff(r_e)
        env = make_env(syms, (r_e, rd_e, rdd_e))
        R_e = evaluate(R_s, env)
        wh_e = evaluate(wh_s, env)
        w_e = unskew(wh_e, tol=tol)
        ah_e = evaluate(ah_s, env)
        a_e = unskew(ah_e, tol=tol)
        return R_e, w_e, a_e

    def project_values(r_e, rd_e, rdd_e):
        # For now, we're ignoring non-uniqueness of rpy.
        return r_e, rd_e, rdd_e

    def calc_angular_velocity_jacobian(r_e):
        T, evaluate, tol = infer_sym_dtype_stuff(r_e)
        env = make_env((r_s,), (r_e,))
        J_e = evaluate(J_s, env)
        return J_e

    return RotationInfo(
        num_rot=len(r_s),
        r0=np.zeros(3),
        calc_values=calc_values,
        project_values=project_values,
        calc_rate_jacobian=make_pinv(calc_angular_velocity_jacobian),
        calc_angular_velocity_jacobian=calc_angular_velocity_jacobian,
    )


@functools.lru_cache
def make_rot_info_quat_sym():
    """Makes Quaternion rotation coordinate info."""
    syms = sym_values_and_1st_and_2nd_derivatives(["w", "x", "y", "z"])
    q_s, qd_s, qdd_s = syms

    q_qd_s = cat(q_s, qd_s)
    qd_qdd_s = cat(qd_s, qdd_s)

    q_norm_squared_s = (q_s**2).sum()
    q_norm_s = np.sqrt(q_norm_squared_s)
    q_full_s = Quaternion_[Expression](wxyz=q_s)

    def remove_q_norm_unit(expr):
        # Assume norm(q)^2 == 1 for expression.
        return drake_sym_replace(expr, q_norm_squared_s, 1.0)

    # Derivatives along normalization (SO(3) projection).
    q_normed_s = q_s / q_norm_s
    qd_normed_s, qdd_normed_s = derivatives_1st_and_2nd(
        q_normed_s, q_s, qd_s, qdd_s
    )
    J_normed_s = Jacobian(q_normed_s, q_s)

    # Nominal SO(3) mapping.
    R_s = RotationMatrix_[Expression](q_full_s).matrix()
    R_s = remove_q_norm_unit(R_s)

    Rd_s, Rdd_s = derivatives_1st_and_2nd(R_s, q_s, qd_s, qdd_s)
    wh_s = Rd_s @ R_s.T
    ah_s = wh_s.T @ wh_s + Rdd_s @ R_s.T
    w_s = unskew(wh_s, tol=None)
    J_s = Jacobian(w_s, qd_s)

    def calc_values(q_e, qd_e, qdd_e):
        T, evaluate, tol = infer_sym_dtype_stuff(q_e)
        env = make_env(syms, (q_e, qd_e, qdd_e))
        R_e = evaluate(R_s, env)
        wh_e = evaluate(wh_s, env)
        w_e = unskew(wh_e, tol=tol)
        ah_e = evaluate(ah_s, env)
        a_e = unskew(ah_e, tol=tol)
        return R_e, w_e, a_e

    def project_values(q_e, qd_e, qdd_e):
        T, evaluate, tol = infer_sym_dtype_stuff(q_e)
        env = make_env(syms, (q_e, qd_e, qdd_e))
        # Normalize input values.
        # TODO(eric.cousineau): What about upper half-sphere?
        q_p = evaluate(q_normed_s, env).reshape((-1,))
        qd_p = evaluate(qd_normed_s, env).reshape((-1,))
        qdd_p = evaluate(qdd_normed_s, env).reshape((-1,))
        return q_p, qd_p, qdd_p

    def calc_angular_velocity_jacobian(q_e):
        T, evaluate, tol = infer_sym_dtype_stuff(q_e)
        env = make_env((q_s,), (q_e,))
        q_normed_e = evaluate(q_normed_s, env).reshape((-1,))
        J_normed_e = evaluate(J_normed_s, env)
        env = make_env((q_s,), (q_normed_e,))
        J_e = evaluate(J_s, env)
        return J_e @ J_normed_e

    return RotationInfo(
        num_rot=len(q_s),
        r0=np.array([1.0, 0.0, 0.0, 0.0]),
        calc_values=calc_values,
        project_values=project_values,
        calc_rate_jacobian=make_pinv(calc_angular_velocity_jacobian),
        calc_angular_velocity_jacobian=calc_angular_velocity_jacobian,
    )


def pinv(A):
    # Useful for symbolics.
    return A.T @ drake_math.inv(A @ A.T)
    # return np.linalg.pinv(A)


def make_pinv(calc):

    def calc_pinv(q):
        M = calc(q)
        Mpinv = pinv(M)
        return Mpinv

    return calc_pinv


def calc_rotational_values(rot_info, r, rd, rdd):
    r, rd, rdd = rot_info.project_values(r, rd, rdd)
    R, w, wd = rot_info.calc_values(r, rd, rdd)
    return (R, w, wd), (r, rd, rdd)


# Reference trajectories.


def min_jerk(s):
    """
    Simple polynomial f(t) for minimum-jerk (zero velocity and acceleration at
    the start and end):

        s=0: f=0, f'=0, f''=0
        s>=1: f=1, f'=0, f''=0
    """
    # TODO(eric.cousineau): Use Drake's polynomial and/or symbolic stuff.
    s = np.clip(s, 0, 1)
    c = [0, 0, 0, 10, -15, 6]
    p = [1, s, s**2, s**3, s**4, s**5]
    pd = [0, 1, 2 * s, 3 * s**2, 4 * s**3, 5 * s**4]
    pdd = [0, 0, 2, 6 * s, 12 * s**2, 20 * s**3]
    f = np.dot(c, p)
    fd = np.dot(c, pd)
    fdd = np.dot(c, pdd)
    return f, fd, fdd


@dc.dataclass
class Sinusoid:
    Ts: np.ndarray
    T0_ratios: np.ndarray = None

    def __post_init__(self):
        self.Ts = np.asarray(self.Ts)
        if self.T0_ratios is not None:
            self.T0_ratios = np.asarray(self.T0_ratios)

    def __call__(self, t):
        ws = 2 * np.pi / self.Ts
        x = ws * t
        if self.T0_ratios is not None:
            x += 2 * np.pi * self.T0_ratios
        dx = ws
        y = np.sin(x)
        yd = dx * np.cos(x)
        ydd = -dx * dx * np.sin(x)
        return y, yd, ydd


@dc.dataclass
class Mult:
    """Second-order multiplication."""
    a: object
    b: object

    def __call__(self, t):
        a, ad, add = self.a(t)
        b, bd, bdd = self.b(t)
        c = a * b
        cd = a * bd + ad * b
        cdd = a * bdd + 2 * (ad * bd) + add * b
        return c, cd, cdd


@dc.dataclass
class So3:
    R: object
    w: object
    wd: object

    def __iter__(self):
        return iter((self.R, self.w, self.wd))

@dc.dataclass
class Coord:
    r: object
    rd: object
    rdd: object

    def __iter__(self):
        return iter((self.r, self.rd, self.rdd))
