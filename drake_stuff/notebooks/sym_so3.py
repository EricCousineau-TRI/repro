# TODO(eric.cousineau): Replace this with `trajectories` and `so2_so3_helpers`.

import dataclasses as dc

import numpy as np

from pydrake.all import Expression, RotationMatrix_, RollPitchYaw_, Variable
import pydrake.symbolic as sym


def skew(r):
    r1, r2, r3 = r
    return np.array([
        [0, -r3, r2],
        [r3, 0, -r1],
        [-r2, r1, 0],
    ])


def unskew(R, *, tol=1e-10):
    if tol is not None:
        # Note: This is non-symbolic.
        dR = R + R.T
        assert np.all(np.max(np.abs(dR)) < tol)
    r1 = R[2, 1]
    r2 = R[0, 2]
    r3 = R[1, 0]
    return np.array([r1, r2, r3])


def differentiate(A, x):
    # Vectorized differentiate for a *scalar* argument.
    # WARNING: `np.vectorize(Expression.Differentiate)` uses NumPy broadcasting
    # rules, so it will decouple derivatives if x is nonscalar :(
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


def cat(*args):
    # Shorthand.
    return np.concatenate(args)


def make_instant_sym(names):
    x = [Variable(name) for name in names]
    # N.B. Putting dot directly after makes things pretty print good in sympy.
    xd = [Variable(f"{name}dot") for name in names]
    xdd = [Variable(f"{name}ddot") for name in names]
    return tuple(np.array(a) for a in (x, xd, xdd))


def aliased_scalar(x, i):
    return x[i:i + 1].reshape(())


def make_aliased_env(instant_sym, instant_values):
    env = {}
    for syms, xs in zip(instant_sym, instant_values):
        for i, sym in enumerate(syms):
            x = aliased_scalar(xs, i)
            env[sym] = x
    return env


@dc.dataclass
class SecondOrderWorkspace:
    # x, xd, xdd
    syms: object
    values: object
    env: object

    @classmethod
    def make(cls, names):
        ndof = len(names)
        syms = make_instant_sym(names)
        values = tuple(np.zeros(ndof) for _ in range(3))
        env = make_aliased_env(syms, values)
        return cls(
            syms=syms,
            values=values,
            env=env,
        )
