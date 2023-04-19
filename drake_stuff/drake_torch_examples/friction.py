"""
Computation for (Coulomb) dry friction.

No fancy things are done (yet), like Stribeck.
Viscous friction is explicitly left out (because it's simply u = -c * v).

Provides simple sigmoid-type regularizers r(s) that have the following
properties:

- r(0) = 0
- r'(0) > 0
- r(inf) = 1, r(-inf) = 1
- r'(inf) = 0, r'(-inf) = 0
- r(1) = m  (a user-supplied constant)

An ideal regularizer should capture dry friction that is could be directly
modeled as `r(s) = sign(s)`, but with sufficient smoothness to make
continuous-time integrators "happy".

These friction models can then be used for friction compensation, which can be
seen as *mostly* feedforward, but more robust (but simple) compenstation
schemes may involve forecasting, e.g. using:
    v_next = v_actual + dt * vd_desired)
"""

import numpy as np


def regularizer_arctan(s, *, m=0.95):
    """
    Regularizer r(s) using an arctan(y) sigmoid.

    Arguments:
        m: Desired value for r(1).
    """
    assert m > 0.0 and m < 1.0
    c = np.tan(np.pi / 2 * m)
    value = 2 / np.pi * np.arctan(c * s)
    return value


def arctanh(y, *, np=np):
    """Inverse of y = tanh(x)"""
    u = -(y + 1) / (y - 1)
    x = np.log(u) / 2
    return x


def regularizer_tanh(s, *, m=0.95, np=np):
    """
    Regularizer r(s) using tanh(x) (hyberbolic tangent) sigmoid.

    Arguments:
        m: Desired value for r(1)
    """
    # TODO(eric.cousineau): This one seems to "simulate better", maybe because
    # gradient is less aggressive at origin than arctan?
    if m is not None:
        assert m > 0.0 and m < 1.0
        c = arctanh(m, np=np)
    else:
        c = 1.0
    value = np.tanh(c * s)
    return value


def calc_joint_dry_friction(v, v0, regularizer, u_max):
    """
    Computes joint-level Coulomb dry friction as:
        u_f = -r(v / v0) * u_max
    Arguments:
        v: Velocity.
        v0:
            Velocity normalization constant. Setting this to a smaller value
            will make models stiffer.
        regularizer:
            Function of the form r(s) per above. Note that the shape of the
            function and value choisen for r(1) can affect stiffness.
        u_max:
            Maximum torque due to dry friction.
    """
    s = v / v0
    u = -regularizer(s) * u_max
    return u
